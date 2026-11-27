"""Evaluate the saved models on the held-out test set and compare them.

For every selected model this computes classification metrics at the configured decision threshold,
the confusion matrix and (optionally) permutation feature importance, then writes them to
``reports/<feature_set>/``:

    metrics.csv / metrics.json       one row per model, sorted by the primary metric
    feature_importance.csv           permutation importance per model and feature
    *.png                            confusion matrices, precision-recall curves, feature importance

Examples:
    python -m src.evaluate
    python -m src.evaluate --set evaluation.threshold=0.3
    python -m src.evaluate --feature-set all --models random_forest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample

from . import plots
from .artifacts import TrainedModel, load_model, load_split, write_json
from .config import PermutationImportanceConfig, PipelineConfig, run_from_command_line

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Test-set results for one model."""

    name: str
    metrics: dict[str, float]
    confusion: np.ndarray
    probabilities: np.ndarray
    importance: pd.DataFrame | None = None

    def as_row(self) -> dict:
        """Flatten metrics and confusion-matrix counts into one table row."""
        tn, fp, fn, tp = self.confusion.ravel()
        return {"model": self.name, **self.metrics, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------
def compute_metrics(
    y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Return the metrics listed in ``config.REPORTED_METRICS``.

    Accuracy, precision, recall and F1 use the thresholded ``predictions``; ROC-AUC and PR-AUC use
    the raw ``probabilities`` so they do not depend on the threshold.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}


def compute_permutation_importance(
    model: TrainedModel,
    X: pd.DataFrame,
    y: pd.Series,
    settings: PermutationImportanceConfig,
    seed: int,
) -> pd.DataFrame:
    """Measure how much the score drops when each feature is randomly shuffled.

    Works the same way for every model type, so importances are comparable across models.
    A stratified subsample of the test set is used when ``max_samples`` is set, to keep it fast.
    """
    if settings.max_samples is not None and settings.max_samples < len(X):
        X, y = resample(
            X, y, n_samples=settings.max_samples, replace=False, stratify=y, random_state=seed
        )
    result = permutation_importance(
        model.pipeline, X, y,
        scoring=settings.scoring, n_repeats=settings.n_repeats, random_state=seed, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "model": model.name,
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(model: TrainedModel, X: pd.DataFrame, y: pd.Series, config: PipelineConfig) -> EvaluationResult:
    """Score one trained model on the test set."""
    missing = sorted(set(model.feature_columns) - set(X.columns))
    if missing:
        raise ValueError(
            f"Test data is missing features that {model.name} was trained on: {missing}. "
            "Were the model and the test data prepared with the same feature set?"
        )
    X = X[model.feature_columns]  # Same column order as during training

    probabilities = model.pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= config.evaluation.threshold).astype(int)
    result = EvaluationResult(
        name=model.name,
        metrics=compute_metrics(y, predictions, probabilities),
        confusion=confusion_matrix(y, predictions, labels=[0, 1]),
        probabilities=probabilities,
    )

    importance_settings = config.evaluation.permutation_importance
    if importance_settings.enabled:
        logger.info("Computing permutation importance for %s", model.name)
        result.importance = compute_permutation_importance(model, X, y, importance_settings, config.seed)
    return result


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------
def build_metrics_table(results: list[EvaluationResult], primary_metric: str) -> pd.DataFrame:
    """Return one row per model, best ``primary_metric`` first."""
    table = pd.DataFrame([result.as_row() for result in results])
    return table.sort_values(primary_metric, ascending=False).reset_index(drop=True)


def save_reports(
    results: list[EvaluationResult], table: pd.DataFrame, y_test: pd.Series, config: PipelineConfig
) -> None:
    """Write the metric tables, feature importance and (optionally) plots to the report folder."""
    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(report_dir / "metrics.csv", index=False)
    write_json(
        {
            "feature_set": config.data.feature_set,
            "threshold": config.evaluation.threshold,
            "primary_metric": config.evaluation.primary_metric,
            "results": table.to_dict(orient="records"),
        },
        report_dir / "metrics.json",
    )

    importances = [result.importance for result in results if result.importance is not None]
    importance_table = pd.concat(importances, ignore_index=True) if importances else None
    if importance_table is not None:
        importance_table.to_csv(report_dir / "feature_importance.csv", index=False)

    if config.evaluation.save_plots:
        plots.plot_confusion_matrices(
            {result.name: result.confusion for result in results}, report_dir / "confusion_matrices.png"
        )
        plots.plot_precision_recall_curves(
            y_test, {result.name: result.probabilities for result in results},
            report_dir / "precision_recall_curves.png",
        )
        if importance_table is not None:
            plots.plot_feature_importance(
                importance_table, config.evaluation.permutation_importance.scoring,
                report_dir / "feature_importance.png",
            )
    logger.info("Reports written to %s", report_dir)


def log_summary(table: pd.DataFrame, results: list[EvaluationResult], config: PipelineConfig) -> None:
    """Print the comparison table and each model's most important features."""
    evaluation = config.evaluation
    logger.info(
        "Test-set results (feature_set=%s, threshold=%.2f, ranked by %s):\n%s",
        config.data.feature_set, evaluation.threshold, evaluation.primary_metric,
        table.to_string(index=False),
    )
    for result in results:
        if result.importance is not None:
            top = result.importance.head(3)
            features = ", ".join(f"{row.feature} ({row.importance_mean:.3f})" for row in top.itertuples())
            logger.info("Top features for %s: %s", result.name, features)


def run(config: PipelineConfig, model_names: list[str] | None = None) -> pd.DataFrame:
    """Evaluate the selected models (all enabled models by default) and return the metrics table."""
    selected = config.select_models(model_names)
    models = [load_model(config.model_dir, name) for name in selected]
    X_test, y_test = load_split(config.test_csv, config.data.target)
    logger.info("Loaded %s: %d rows, %.1f%% positive", config.test_csv, len(X_test), 100 * y_test.mean())

    results = [evaluate_model(model, X_test, y_test, config) for model in models]
    table = build_metrics_table(results, config.evaluation.primary_metric)
    save_reports(results, table, y_test, config)
    log_summary(table, results, config)
    return table


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``python -m src.evaluate [options]``."""
    run_from_command_line(__doc__, run, argv)


if __name__ == "__main__":
    main()

"""Train, tune and save every model configured in src/config.yaml.

Reads the cleaned training split produced by ``src.prepare_data``, optionally tunes each model's
hyperparameters with stratified cross-validation, refits the best version on the full training set
and saves it to ``saved_model/<feature_set>/<model_name>.joblib``.

Examples:
    python -m src.train
    python -m src.train --models random_forest --set tuning.method=grid
    python -m src.train --set tuning.enabled=false --set models.random_forest.params.n_estimators=500
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import yaml

from .artifacts import TrainedModel, load_split, save_model, write_json
from .config import ModelConfig, PipelineConfig, run_from_command_line
from .model_factory import build_pipeline, build_search, strip_step_prefix

logger = logging.getLogger(__name__)


def train_model(
    name: str, model_config: ModelConfig, config: PipelineConfig, X: pd.DataFrame, y: pd.Series
) -> TrainedModel:
    """Fit one model, tuning it first when tuning is enabled and it has a search space."""
    pipeline = build_pipeline(model_config, config.seed)
    tune = config.tuning.enabled and bool(model_config.search_space)
    best_params: dict = {}
    cv_score = None

    start = time.perf_counter()
    if tune:
        search = build_search(pipeline, model_config.search_space, config.tuning, config.seed)
        logger.info(
            "Tuning %s with %s search (%d-fold CV, scoring=%s)",
            name, config.tuning.method, config.tuning.cv_folds, config.tuning.scoring,
        )
        search.fit(X, y)
        pipeline = search.best_estimator_
        best_params = strip_step_prefix(search.best_params_)
        cv_score = float(search.best_score_)
    else:
        logger.info("Training %s with fixed hyperparameters (no tuning)", name)
        pipeline.fit(X, y)
    elapsed = time.perf_counter() - start

    return TrainedModel(
        name=name,
        pipeline=pipeline,
        feature_columns=list(X.columns),
        feature_set=config.data.feature_set,
        estimator=model_config.estimator,
        best_params=best_params,
        cv_scoring=config.tuning.scoring if tune else None,
        cv_score=cv_score,
        training_seconds=elapsed,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def log_result(model: TrainedModel) -> None:
    """Log a one-line summary of a trained model."""
    score = f"CV {model.cv_scoring}={model.cv_score:.4f}" if model.cv_score is not None else "not tuned"
    logger.info("Finished %s in %.1fs (%s)", model.name, model.training_seconds, score)
    if model.best_params:
        logger.info("  Best params: %s", model.best_params)


def save_run_metadata(config: PipelineConfig, models: list[TrainedModel]) -> None:
    """Save the training summary and the exact configuration used, for reproducibility.

    Entries for models that were not retrained in this run are kept in the summary.
    """
    summary_path = config.model_dir / "training_summary.json"
    summaries = {}
    if summary_path.is_file():
        summaries = {entry["name"]: entry for entry in json.loads(summary_path.read_text("utf-8"))}
    summaries.update({model.name: model.summary() for model in models})
    write_json(list(summaries.values()), summary_path)

    with (config.model_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False)


def run(config: PipelineConfig, model_names: list[str] | None = None) -> list[TrainedModel]:
    """Train and save the selected models (all enabled models by default)."""
    selected = config.select_models(model_names)
    X, y = load_split(config.train_csv, config.data.target)
    logger.info(
        "Loaded %s: %d rows, %d features, %.1f%% positive",
        config.train_csv, len(X), X.shape[1], 100 * y.mean(),
    )

    trained = []
    for name, model_config in selected.items():
        model = train_model(name, model_config, config, X, y)
        path = save_model(model, config.model_dir)
        log_result(model)
        logger.info("  Saved to %s", path)
        trained.append(model)

    save_run_metadata(config, trained)
    return trained


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``python -m src.train [options]``."""
    run_from_command_line(__doc__, run, argv)


if __name__ == "__main__":
    main()

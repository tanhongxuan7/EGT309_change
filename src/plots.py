"""Charts that compare the evaluated models side by side."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to files only, so this also works in Docker without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

CLASS_LABELS = ("No maintenance", "Maintenance")


def _save(figure: plt.Figure, path: Path) -> None:
    """Write ``figure`` to ``path`` and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion_matrices(matrices: dict[str, np.ndarray], path: Path) -> None:
    """Draw one confusion matrix per model next to each other."""
    figure, axes = plt.subplots(1, len(matrices), figsize=(4.5 * len(matrices), 4), squeeze=False)
    for ax, (name, matrix) in zip(axes[0], matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_LABELS).plot(
            ax=ax, cmap="Blues", values_format=",d", colorbar=False
        )
        ax.set_title(name)
    _save(figure, path)


def plot_precision_recall_curves(
    y_true: pd.Series, probabilities: dict[str, np.ndarray], path: Path
) -> None:
    """Overlay the precision-recall curve of every model on one chart."""
    figure, ax = plt.subplots(figsize=(7, 5))
    for name, scores in probabilities.items():
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=name, ax=ax)
    ax.axhline(y_true.mean(), color="grey", linestyle="--", label="No-skill baseline")
    ax.set_title("Precision-recall curves (test set)")
    ax.legend(loc="lower left")
    _save(figure, path)


def plot_feature_importance(importance: pd.DataFrame, scoring: str, path: Path) -> None:
    """Draw a horizontal bar chart of permutation importance for each model.

    ``importance`` must have the columns ``model``, ``feature``, ``importance_mean`` and ``importance_std``.
    """
    models = importance["model"].unique()
    figure, axes = plt.subplots(
        1, len(models), figsize=(5 * len(models), 0.4 * importance["feature"].nunique() + 1.5),
        squeeze=False,
    )
    for ax, model in zip(axes[0], models):
        rows = importance[importance["model"] == model].sort_values("importance_mean")
        ax.barh(rows["feature"], rows["importance_mean"], xerr=rows["importance_std"], color="tab:blue")
        ax.set_title(model)
        ax.set_xlabel(f"Drop in {scoring} when shuffled")
    _save(figure, path)

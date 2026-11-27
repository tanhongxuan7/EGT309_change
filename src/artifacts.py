"""Read the prepared (already cleaned) datasets and save/load trained model artifacts.

Shared by ``src.train`` and ``src.evaluate`` so both steps read data and models the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline

MODEL_SUFFIX = ".joblib"


# --------------------------------------------------------------------------------------------
# Prepared datasets
# --------------------------------------------------------------------------------------------
def load_split(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a prepared train/test CSV and split it into features ``X`` and target ``y``.

    The CSV is expected to be already cleaned by the data-preparation step, so this only checks
    that assumption (target present, binary labels, numeric features, no missing values).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared data not found: {path}\n"
            "Run the data preparation step first, e.g. `python -m src.prepare_data`."
        )
    data = pd.read_csv(path)

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing from {path}.")
    labels = set(data[target].unique())
    if not labels <= {0, 1} or len(labels) < 2:
        raise ValueError(f"'{target}' in {path} must contain both labels 0 and 1; found {sorted(labels)}.")

    features = data.drop(columns=target)
    non_numeric = [column for column in features.columns if not is_numeric_dtype(features[column])]
    if non_numeric:
        raise ValueError(f"Features must be numeric; non-numeric columns in {path}: {non_numeric}.")
    missing = features.columns[features.isna().any()].tolist()
    if missing:
        raise ValueError(f"Expected cleaned data, but {path} has missing values in: {missing}.")

    return features, data[target].astype(int)


# --------------------------------------------------------------------------------------------
# Trained models
# --------------------------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """A fitted pipeline plus the metadata needed to evaluate and reproduce it."""

    name: str
    pipeline: Pipeline
    feature_columns: list[str]
    feature_set: str
    estimator: str
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_scoring: str | None = None
    cv_score: float | None = None
    training_seconds: float = 0.0
    trained_at: str = ""

    def summary(self) -> dict[str, Any]:
        """Return the metadata (everything except the fitted pipeline) as JSON-friendly values."""
        return {
            "name": self.name,
            "estimator": self.estimator,
            "feature_set": self.feature_set,
            "feature_columns": self.feature_columns,
            "best_params": self.best_params,
            "cv_scoring": self.cv_scoring,
            "cv_score": self.cv_score,
            "training_seconds": round(self.training_seconds, 2),
            "trained_at": self.trained_at,
        }


def model_path(model_dir: Path, name: str) -> Path:
    """Return the file used to store the model called ``name``."""
    return model_dir / f"{name}{MODEL_SUFFIX}"


def save_model(model: TrainedModel, model_dir: Path) -> Path:
    """Save ``model`` to ``model_dir`` and return the file path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(model_dir, model.name)
    joblib.dump(model, path)
    return path


def load_model(model_dir: Path, name: str) -> TrainedModel:
    """Load the model called ``name`` from ``model_dir``."""
    path = model_path(model_dir, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}\nTrain it first, e.g. "
            f"`python -m src.train --feature-set {model_dir.name} --models {name}`."
        )
    model = joblib.load(path)
    if not isinstance(model, TrainedModel):
        raise TypeError(f"{path} does not contain a TrainedModel.")
    return model


# --------------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------------
def write_json(data: Any, path: Path) -> None:
    """Write ``data`` to ``path`` as indented JSON, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)

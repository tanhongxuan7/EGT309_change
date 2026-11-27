"""Build scikit-learn model pipelines and hyperparameter searches from configuration.

Models are referenced in the config by their import path (e.g.
``sklearn.ensemble.RandomForestClassifier``), so new models can be added by editing
``src/config.yaml`` only.
"""

from __future__ import annotations

import importlib
import inspect
import math
from typing import Any

from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig, TuningConfig

MODEL_STEP = "model"
SCALER_STEP = "scaler"


def import_estimator(dotted_path: str) -> type:
    """Import an estimator class from a path such as ``sklearn.linear_model.LogisticRegression``."""
    module_name, _, class_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Estimator must be a full import path, e.g. 'sklearn.svm.SVC'; got {dotted_path!r}.")
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot import estimator {dotted_path!r}: {error}") from error


def _accepts_random_state(estimator_class: type) -> bool:
    """Return True if the estimator's constructor has a ``random_state`` argument."""
    return "random_state" in inspect.signature(estimator_class).parameters


def build_pipeline(model_config: ModelConfig, seed: int) -> Pipeline:
    """Create an unfitted pipeline: optional standard scaling followed by the configured classifier.

    ``seed`` is passed as ``random_state`` unless the config already sets one, so runs are reproducible.
    """
    estimator_class = import_estimator(model_config.estimator)
    params = dict(model_config.params)
    if _accepts_random_state(estimator_class):
        params.setdefault("random_state", seed)

    try:
        estimator = estimator_class(**params)
    except TypeError as error:
        raise ValueError(f"Invalid params for {model_config.estimator}: {error}") from error
    if not is_classifier(estimator):
        raise ValueError(f"{model_config.estimator} is not a classifier.")

    steps = [(SCALER_STEP, StandardScaler())] if model_config.scale_features else []
    steps.append((MODEL_STEP, estimator))
    return Pipeline(steps)


def build_search(
    pipeline: Pipeline, search_space: dict[str, list[Any]], tuning: TuningConfig, seed: int
) -> GridSearchCV | RandomizedSearchCV:
    """Wrap ``pipeline`` in a stratified cross-validated grid or random search.

    Search-space names are given without a prefix in the config; this adds the pipeline step prefix.
    """
    param_grid = {f"{MODEL_STEP}__{name}": values for name, values in search_space.items()}
    shared = {
        "scoring": tuning.scoring,
        "cv": StratifiedKFold(n_splits=tuning.cv_folds, shuffle=True, random_state=seed),
        "n_jobs": tuning.n_jobs,
        "refit": True,
    }
    if tuning.method == "grid":
        return GridSearchCV(pipeline, param_grid, **shared)

    # Never sample more candidates than exist (avoids a scikit-learn warning on small spaces).
    n_candidates = math.prod(len(values) for values in param_grid.values())
    n_iter = min(tuning.n_iter, n_candidates)
    return RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, random_state=seed, **shared)


def strip_step_prefix(params: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{'model__C': 1.0}`` back into ``{'C': 1.0}`` for readable reports."""
    prefix = f"{MODEL_STEP}__"
    return {name.removeprefix(prefix): value for name, value in params.items()}

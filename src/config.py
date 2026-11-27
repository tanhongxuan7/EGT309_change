"""Load and validate the training/evaluation configuration (src/config.yaml).

The YAML file is turned into typed, read-only dataclasses so that every other module accesses
settings as attributes (``config.tuning.cv_folds``) and typos or invalid values fail early with a
clear message instead of deep inside scikit-learn.
"""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from sklearn.metrics import get_scorer_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"

FEATURE_SETS = ("sensor", "all")
SEARCH_METHODS = ("grid", "random")
REPORTED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


# --------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------
def _require_choice(name: str, value: Any, choices: Iterable[Any]) -> None:
    """Raise if ``value`` is not one of ``choices``."""
    choices = tuple(choices)
    if value not in choices:
        raise ValueError(f"'{name}' must be one of {', '.join(map(str, choices))}; got {value!r}.")


def _require_scorer(name: str, scoring: str) -> None:
    """Raise if ``scoring`` is not a scikit-learn scorer name."""
    if scoring not in get_scorer_names():
        raise ValueError(f"'{name}' is not a valid scikit-learn scorer: {scoring!r}.")


def _require_at_least(name: str, value: Any, minimum: int) -> None:
    """Raise if ``value`` is not an integer greater than or equal to ``minimum``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"'{name}' must be an integer >= {minimum}; got {value!r}.")


def resolve_path(path: str | Path) -> Path:
    """Return ``path`` as an absolute path, treating relative paths as relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------------------------
# Configuration sections
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Location of the cleaned train/test CSVs produced by ``src.prepare_data``."""

    feature_set: str = "sensor"
    prepared_dir: str = "data/prepared"
    target: str = "maintenance_required"

    def __post_init__(self) -> None:
        _require_choice("data.feature_set", self.feature_set, FEATURE_SETS)


@dataclass(frozen=True)
class OutputConfig:
    """Folders where trained models and evaluation reports are written."""

    model_dir: str = "saved_model"
    report_dir: str = "reports"


@dataclass(frozen=True)
class TuningConfig:
    """Cross-validated hyperparameter search settings shared by all models."""

    enabled: bool = True
    method: str = "random"
    n_iter: int = 10
    cv_folds: int = 5
    scoring: str = "average_precision"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        _require_choice("tuning.method", self.method, SEARCH_METHODS)
        _require_at_least("tuning.n_iter", self.n_iter, 1)
        _require_at_least("tuning.cv_folds", self.cv_folds, 2)
        _require_scorer("tuning.scoring", self.scoring)


@dataclass(frozen=True)
class ModelConfig:
    """One candidate model: which estimator to build, its fixed parameters and its search space."""

    estimator: str
    enabled: bool = True
    scale_features: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, values in self.search_space.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"search_space '{name}' must be a non-empty list; got {values!r}.")


@dataclass(frozen=True)
class PermutationImportanceConfig:
    """Settings for model-agnostic feature importance computed on the test set."""

    enabled: bool = True
    n_repeats: int = 5
    scoring: str = "average_precision"
    max_samples: int | None = 5000

    def __post_init__(self) -> None:
        _require_at_least("evaluation.permutation_importance.n_repeats", self.n_repeats, 1)
        _require_scorer("evaluation.permutation_importance.scoring", self.scoring)
        if self.max_samples is not None:
            _require_at_least("evaluation.permutation_importance.max_samples", self.max_samples, 1)


@dataclass(frozen=True)
class EvaluationConfig:
    """How trained models are scored on the held-out test set."""

    threshold: float = 0.5
    primary_metric: str = "recall"
    save_plots: bool = True
    permutation_importance: PermutationImportanceConfig = field(
        default_factory=PermutationImportanceConfig
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"'evaluation.threshold' must be between 0 and 1; got {self.threshold}.")
        _require_choice("evaluation.primary_metric", self.primary_metric, REPORTED_METRICS)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete configuration for the training and evaluation steps."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        _require_at_least("seed", self.seed, 0)

    # Paths are split by feature set so "sensor" and "all" runs never overwrite each other.
    @property
    def train_csv(self) -> Path:
        return resolve_path(self.data.prepared_dir) / self.data.feature_set / "train.csv"

    @property
    def test_csv(self) -> Path:
        return resolve_path(self.data.prepared_dir) / self.data.feature_set / "test.csv"

    @property
    def model_dir(self) -> Path:
        return resolve_path(self.output.model_dir) / self.data.feature_set

    @property
    def report_dir(self) -> Path:
        return resolve_path(self.output.report_dir) / self.data.feature_set

    def select_models(self, names: Iterable[str] | None = None) -> dict[str, ModelConfig]:
        """Return the requested models, or every enabled model when ``names`` is empty.

        Models named explicitly are returned even if they are disabled in the config.
        """
        if not names:
            selected = {name: model for name, model in self.models.items() if model.enabled}
        else:
            unknown = sorted(set(names) - set(self.models))
            if unknown:
                raise ValueError(
                    f"Unknown model(s): {', '.join(unknown)}. "
                    f"Configured models: {', '.join(self.models)}."
                )
            selected = {name: self.models[name] for name in names}
        if not selected:
            raise ValueError("No models selected: enable at least one model in the config.")
        return selected

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as plain Python types (e.g. to save alongside a model)."""
        return asdict(self)


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def _build_section(cls: type, raw: Any, section: str) -> Any:
    """Create dataclass ``cls`` from a mapping, rejecting unknown or missing keys with a clear error."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError(f"'{section}' must be a mapping; got {type(raw).__name__}.")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{section}': {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}."
        )
    try:
        return cls(**raw)
    except TypeError as error:  # Raised for missing required keys
        raise ValueError(f"Invalid '{section}' section: {error}") from error


def build_config(raw: dict[str, Any]) -> PipelineConfig:
    """Convert a raw configuration mapping (e.g. parsed YAML) into a validated ``PipelineConfig``."""
    raw = dict(raw or {})
    evaluation = dict(raw.pop("evaluation", None) or {})
    evaluation["permutation_importance"] = _build_section(
        PermutationImportanceConfig,
        evaluation.get("permutation_importance"),
        "evaluation.permutation_importance",
    )
    models = raw.pop("models", None) or {}
    if not isinstance(models, dict):
        raise ValueError("'models' must be a mapping of model name to model settings.")

    sections = {
        "data": _build_section(DataConfig, raw.pop("data", None), "data"),
        "output": _build_section(OutputConfig, raw.pop("output", None), "output"),
        "tuning": _build_section(TuningConfig, raw.pop("tuning", None), "tuning"),
        "evaluation": _build_section(EvaluationConfig, evaluation, "evaluation"),
        "models": {
            name: _build_section(ModelConfig, settings, f"models.{name}")
            for name, settings in models.items()
        },
    }
    return _build_section(PipelineConfig, {**raw, **sections}, "config")


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with ``dotted.key=value`` overrides applied.

    Values are parsed as YAML, so ``true``, ``0.3``, ``null`` and ``[1, 2]`` keep their types.
    """
    updated = copy.deepcopy(raw)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Override {override!r} must look like key.subkey=value.")
        *parents, leaf = key.strip().split(".")
        node = updated
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: '{part}' is not a section.")
        node[leaf] = yaml.safe_load(value)
    return updated


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH, overrides: Iterable[str] = ()
) -> PipelineConfig:
    """Read the YAML file at ``path``, apply command-line overrides and validate the result."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return build_config(apply_overrides(raw, overrides))


# --------------------------------------------------------------------------------------------
# Shared command-line helpers for src.train and src.evaluate
# --------------------------------------------------------------------------------------------
def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that both the training and the evaluation scripts accept."""
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="YAML configuration file (default: src/config.yaml).",
    )
    parser.add_argument(
        "--feature-set", choices=FEATURE_SETS,
        help="Shortcut for --set data.feature_set=<value>.",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME",
        help="Only use these models (default: every model enabled in the config).",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set models.random_forest.params.n_estimators=500. "
             "Can be repeated.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Load the configuration described by the options from ``add_config_arguments``."""
    overrides = list(args.overrides)
    if args.feature_set:
        overrides.append(f"data.feature_set={args.feature_set}")
    return load_config(args.config, overrides)


def configure_logging(level: str = "INFO") -> None:
    """Send timestamped log messages to the console."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_from_command_line(
    description: str,
    run: Callable[[PipelineConfig, list[str] | None], Any],
    argv: list[str] | None = None,
) -> None:
    """Parse the shared options, load the config and call ``run(config, model_names)``.

    Configuration and missing-file errors are shown as a short message instead of a traceback.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        run(config_from_args(args), args.models)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

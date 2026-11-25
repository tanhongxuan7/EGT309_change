"""Build model features using statistics learned from training data."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .ingest import DERIVED, SENSORS


class ReadingFeatures(TransformerMixin, BaseEstimator):
    """Transform raw readings into sensor or comparison features."""

    def __init__(self, feature_set: str = "sensor"):
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y=None):
        if self.feature_set not in ("sensor", "all"):
            raise ValueError("feature_set must be 'sensor' or 'all'.")
        median = X["temperature"].median()
        if pd.isna(median):
            raise ValueError("Training data must contain at least one temperature reading.")

        self.failure_encoder_ = None
        if self.feature_set == "all":
            self.failure_encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=int
            ).fit(X[["failure_type"]])
        self.temperature_median_ = median
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "temperature_median_")
        features = X.loc[:, list(SENSORS)].copy()
        features["temperature_missing"] = features["temperature"].isna().astype(int)
        features["temperature"] = features["temperature"].fillna(self.temperature_median_)
        features["vibration"] = features["vibration"].abs()
        features["temp_above_90"] = (features["temperature"] > 90).astype(int)
        features["vibration_above_80"] = (features["vibration"] > 80).astype(int)

        if self.failure_encoder_ is not None:
            for column in DERIVED:
                if column != "failure_type":
                    features[column] = X[column]
            encoded = self.failure_encoder_.transform(X[["failure_type"]])
            names = self.failure_encoder_.get_feature_names_out()
            features[names] = encoded
        return features

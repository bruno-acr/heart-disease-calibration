"""Base classifier: preprocessing + Random Forest in a single leakage-safe pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, NUMERIC


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def make_model(seed: int = 42, n_estimators: int = 300) -> Pipeline:
    """Random Forest with hyperparameters fixed a priori (no tuning on the evaluation data)."""
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=3, random_state=seed, n_jobs=1)),
    ])

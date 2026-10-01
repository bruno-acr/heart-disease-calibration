"""Loading and validation of the UCI Heart Disease (Cleveland) dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed.cleveland.data"

# SHA-256 of the copy of `processed.cleveland.data` shipped in `data/`.
# It guards against silent changes to the file; it is not a checksum published by UCI.
SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Recorded sex (0 = female, 1 = male)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar > 120 mg/dL (0/1)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (0/1)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak-exercise ST segment (1-3)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium test (3 = normal, 6 = fixed defect, 7 = reversible defect)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path | str | None = None, verify: bool = True) -> pd.DataFrame:
    """Read the raw file; `?` marks missing values."""
    path = Path(path) if path is not None else DATA_PATH
    if verify and sha256(path) != SHA256:
        raise ValueError(f"Unexpected SHA-256 for {path}; the data file has changed.")
    raw = pd.read_csv(path, names=COLUMNS, na_values="?")
    if raw.shape != (303, 14) or raw["num"].isna().any():
        raise ValueError(f"Unexpected dataset shape or missing outcome: {raw.shape}")
    return raw


def load_cleveland(path: Path | str | None = None, verify: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictors `X` and the binary outcome `y` (1 = disease present, `num` > 0)."""
    raw = load_raw(path, verify)
    X = raw.drop(columns="num")
    y = (raw["num"] > 0).astype(int).rename("disease")
    return X, y

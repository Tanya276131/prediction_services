"""
Data preprocessing for the Telco Customer Churn dataset.

Turns the raw IBM Telco CSV into a clean, model-ready dataframe and
persists the fitted preprocessing artifacts (encoders, column list) so
the exact same transform can be replayed at inference time in the API.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DIR = Path("data/processed")
ARTIFACT_DIR = Path("models")

TARGET_COL = "Churn"
ID_COL = "customerID"

# Columns that are genuinely categorical/binary strings in this dataset.
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is stored as a string with some blank entries for
    # brand-new customers (tenure == 0); coerce and impute with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop(columns=[ID_COL])
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def fit_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categoricals, fitting fresh encoders. Returns the
    encoded dataframe plus a dict of fitted LabelEncoders (for reuse at
    inference time)."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def save_artifacts(df: pd.DataFrame, encoders: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DIR / "churn_processed.csv", index=False)
    joblib.dump(encoders, ARTIFACT_DIR / "label_encoders.joblib")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    with open(ARTIFACT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)


def run_pipeline() -> pd.DataFrame:
    raw = load_raw()
    cleaned = clean(raw)
    encoded, encoders = fit_transform(cleaned)
    save_artifacts(encoded, encoders)
    print(f"Processed {len(encoded)} rows, {len(encoded.columns) - 1} features.")
    print(f"Churn rate: {encoded[TARGET_COL].mean():.3%}")
    return encoded


if __name__ == "__main__":
    run_pipeline()

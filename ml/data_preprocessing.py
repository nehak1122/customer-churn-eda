"""Data loading and preprocessing for the Telco Customer Churn dataset.

Keeps a raw -> encoded transform as a reusable function so the
intervention simulator can modify human-readable features (e.g. change
Contract from "Month-to-month" to "One year") and re-encode them the
same way the model was trained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"

# Yes/No style binary columns (encoded 1/0)
BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Multi-category columns (one-hot encoded)
MULTI_COLS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the raw CSV (fixes blank TotalCharges for tenure-0 rows)."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def encode(df_raw: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Encode a raw dataframe into the model feature matrix.

    When `feature_columns` is given (from training), the output is
    reindexed to exactly those columns so single-row what-if frames
    keep every one-hot column even if a category is absent.
    """
    df = df_raw.drop(columns=[ID_COL, TARGET], errors="ignore").copy()
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=False, dtype=int)
    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def prepare_datasets(test_size: float = 0.2, random_state: int = 42):
    """Return (X_train, X_test, y_train, y_test, feature_columns, df_raw)."""
    df_raw = load_raw()
    y = (df_raw[TARGET] == "Yes").astype(int)
    X = encode(df_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns), df_raw


def class_balance(y: pd.Series) -> dict:
    counts = y.value_counts()
    return {
        "stayed": int(counts.get(0, 0)),
        "churned": int(counts.get(1, 0)),
        "churn_rate": round(float(y.mean()) * 100, 2),
    }

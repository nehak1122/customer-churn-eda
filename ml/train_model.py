"""Model training addressing Gap 1 from the 20-paper literature review:
class imbalance in churn data is rarely handled carefully.

Trains Logistic Regression, Random Forest and XGBoost twice each -
once on the raw imbalanced training set and once on a SMOTE-balanced
training set - and evaluates every variant on the same untouched test
set, so the effect of imbalance handling is measured, not assumed.

Run from the repo root:  python -m ml.train_model
"""

import json
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from ml.data_preprocessing import class_balance, prepare_datasets

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def build_models(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=4000, solver="liblinear", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, feature_columns, _ = prepare_datasets()

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    results = {}
    trained = {}
    for balanced, (Xt, yt) in [(False, (X_train, y_train)), (True, (X_bal, y_bal))]:
        for name, model in build_models().items():
            model.fit(Xt, yt)
            key = f"{name}{' + SMOTE' if balanced else ''}"
            results[key] = evaluate(model, X_test, y_test)
            trained[key] = model
            print(f"{key:28s} {results[key]}")

    # Best model by F1 on the churn class (the metric imbalance hurts most)
    best_key = max(results, key=lambda k: results[k]["f1"])
    best_model = trained[best_key]
    print(f"\nBest model by F1: {best_key}")

    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    metadata = {
        "best_model": best_key,
        "feature_columns": feature_columns,
        "results": results,
        "train_balance_before_smote": class_balance(y_train),
        "train_balance_after_smote": class_balance(y_bal),
        "test_size": len(y_test),
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

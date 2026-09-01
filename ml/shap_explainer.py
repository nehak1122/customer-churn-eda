"""SHAP explainability addressing Gap 2 from the literature review:
most churn studies predict who will leave without explaining why in a
way business teams can act on.

Provides global feature importance and per-customer explanations for
the trained best model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def load_model():
    return joblib.load(ARTIFACTS_DIR / "best_model.joblib")


def make_explainer(model, X_background: pd.DataFrame):
    """TreeExplainer for tree models, LinearExplainer for linear models
    (both fast/exact), generic Explainer as a last resort."""
    try:
        return shap.TreeExplainer(model)
    except Exception:
        pass
    sample = shap.sample(X_background, 200, random_state=42)
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, shap.maskers.Independent(sample))
    return shap.Explainer(model.predict_proba, sample)


def shap_values_for(explainer, X: pd.DataFrame) -> np.ndarray:
    """Return a 2-D (rows x features) array of SHAP values for the churn class."""
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older API: [class0, class1]
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, 1]
    return values


def global_importance(explainer, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    values = shap_values_for(explainer, X)
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return importance.head(top_n).reset_index(drop=True)


def explain_customer(explainer, X_row: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top features pushing one customer toward or away from churning."""
    values = shap_values_for(explainer, X_row)[0]
    contrib = pd.DataFrame(
        {
            "feature": X_row.columns,
            "value": X_row.iloc[0].values,
            "shap_value": values,
        }
    )
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")
    return contrib.head(top_n).reset_index(drop=True)

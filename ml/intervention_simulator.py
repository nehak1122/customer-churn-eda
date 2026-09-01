"""What-if intervention simulator addressing Gap 3 - the biggest gap
found across the 20 reviewed papers (2015-2026): almost no study tests
whether acting on the model's findings would actually reduce churn.

Interventions are applied to the raw, human-readable customer record
(e.g. Contract -> "One year"), the record is re-encoded exactly as at
training time, and the churn probability is re-scored, so the output is
"predicted risk before vs after the fix".
"""

import pandas as pd

from ml.data_preprocessing import encode

# Business levers a retention team can actually pull, mapped to raw columns.
INTERVENTIONS = {
    "Upgrade to One-year contract": {"Contract": "One year"},
    "Upgrade to Two-year contract": {"Contract": "Two year"},
    "Add Tech Support": {"TechSupport": "Yes"},
    "Add Online Security": {"OnlineSecurity": "Yes"},
    "Move to automatic bank transfer payment": {"PaymentMethod": "Bank transfer (automatic)"},
    "10% discount on monthly charges": {"__discount__": 0.10},
    "20% discount on monthly charges": {"__discount__": 0.20},
}


def apply_intervention(raw_row: pd.DataFrame, changes: dict) -> pd.DataFrame:
    modified = raw_row.copy()
    for col, value in changes.items():
        if col == "__discount__":
            modified["MonthlyCharges"] = modified["MonthlyCharges"] * (1 - value)
        else:
            modified[col] = value
    return modified


def churn_probability(model, raw_rows: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    X = encode(raw_rows, feature_columns=feature_columns)
    return pd.Series(model.predict_proba(X)[:, 1], index=raw_rows.index)


def simulate_customer(model, raw_row: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Score every intervention for one customer. Returns before/after/delta."""
    base = float(churn_probability(model, raw_row, feature_columns).iloc[0])
    records = []
    for name, changes in INTERVENTIONS.items():
        modified = apply_intervention(raw_row, changes)
        after = float(churn_probability(model, modified, feature_columns).iloc[0])
        records.append(
            {
                "intervention": name,
                "risk_before": round(base, 4),
                "risk_after": round(after, 4),
                "risk_reduction": round(base - after, 4),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("risk_reduction", ascending=False)
        .reset_index(drop=True)
    )


def simulate_cohort(
    model,
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    risk_threshold: float = 0.5,
) -> pd.DataFrame:
    """Apply each intervention to every at-risk customer and measure the
    average predicted-risk reduction across the cohort - the paper-level
    evidence that 'fixing the problem' works."""
    base = churn_probability(model, raw_df, feature_columns)
    at_risk = raw_df[base >= risk_threshold]
    if at_risk.empty:
        return pd.DataFrame()
    base_risk = base[base >= risk_threshold]
    records = []
    for name, changes in INTERVENTIONS.items():
        modified = apply_intervention(at_risk, changes)
        after = churn_probability(model, modified, feature_columns)
        records.append(
            {
                "intervention": name,
                "customers_at_risk": len(at_risk),
                "avg_risk_before": round(float(base_risk.mean()), 4),
                "avg_risk_after": round(float(after.mean()), 4),
                "avg_risk_reduction": round(float((base_risk - after).mean()), 4),
                "customers_moved_below_threshold": int((after < risk_threshold).sum()),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("avg_risk_reduction", ascending=False)
        .reset_index(drop=True)
    )

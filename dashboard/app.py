"""Customer Churn Gap-Analysis Dashboard.

Streamlit app implementing the three gaps identified in the review of
20 churn research papers (2015-2026):
  1. Class imbalance handling (SMOTE) - measured, not assumed
  2. Explainability (SHAP) - global and per-customer
  3. What-if intervention simulation - does fixing the problem work?

Run from the repo root:  streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.data_preprocessing import encode, load_raw  # noqa: E402
from ml.intervention_simulator import (  # noqa: E402
    churn_probability,
    simulate_cohort,
    simulate_customer,
)
from ml.shap_explainer import (  # noqa: E402
    explain_customer,
    global_importance,
    make_explainer,
)

ARTIFACTS = ROOT / "ml" / "artifacts"

st.set_page_config(page_title="Churn Gap-Analysis Dashboard", page_icon="📉", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    with open(ARTIFACTS / "metrics.json") as f:
        meta = json.load(f)
    df_raw = load_raw()
    X_all = encode(df_raw, feature_columns=meta["feature_columns"])
    explainer = make_explainer(model, X_all)
    return model, meta, df_raw, X_all, explainer


try:
    model, meta, df_raw, X_all, explainer = load_assets()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python -m ml.train_model` first.")
    st.stop()


@st.cache_data
def cached_global_importance(top_n: int = 15) -> pd.DataFrame:
    sample = X_all.sample(min(800, len(X_all)), random_state=42)
    return global_importance(explainer, sample, top_n=top_n)


@st.cache_data
def cached_cohort_sim(threshold: float) -> pd.DataFrame:
    return simulate_cohort(model, df_raw, meta["feature_columns"], risk_threshold=threshold)

risk_all = churn_probability(model, df_raw, meta["feature_columns"])

st.title("📉 Customer Churn — Gap-Analysis Dashboard")
st.caption(
    "Implements the three gaps found across 20 research papers (2015–2026): "
    "class imbalance handling, SHAP explainability, and what-if intervention testing."
)

tab_overview, tab_gap1, tab_gap2, tab_gap3 = st.tabs(
    [
        "📊 Overview",
        "⚖️ Gap 1 · Imbalance (SMOTE)",
        "🔍 Gap 2 · Explainability (SHAP)",
        "🧪 Gap 3 · What-If Simulator",
    ]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    churn_rate = (df_raw["Churn"] == "Yes").mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df_raw):,}")
    c2.metric("Churn rate", f"{churn_rate:.1f}%")
    c3.metric("Best model", meta["best_model"])
    c4.metric("At-risk customers (p ≥ 0.5)", f"{int((risk_all >= 0.5).sum()):,}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            df_raw.assign(risk=risk_all),
            x="risk",
            nbins=40,
            title="Predicted churn-risk distribution (all customers)",
            labels={"risk": "Predicted churn probability"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_contract = (
            df_raw.assign(churned=(df_raw["Churn"] == "Yes").astype(int))
            .groupby("Contract")["churned"]
            .mean()
            .mul(100)
            .reset_index()
        )
        fig = px.bar(
            by_contract,
            x="Contract",
            y="churned",
            title="Actual churn rate by contract type",
            labels={"churned": "Churn rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This dashboard extends the original EDA (see `Customer_Churn_EDA.ipynb`). "
        "Each of the next three tabs closes one gap identified in the literature review — "
        "see `GAP_ANALYSIS.md` for the paper-by-paper reasoning."
    )

# ---------------------------------------------------------------- Gap 1
with tab_gap1:
    st.subheader("Gap 1 — Class imbalance is rarely handled carefully")
    st.markdown(
        "Only about a quarter of customers churn, so models trained on raw data look "
        "accurate while missing most actual churners. Following Amin et al. (2016) and "
        "Zhu et al. (2017), every model was trained **with and without SMOTE** and "
        "evaluated on the same untouched test set."
    )

    bal_before = meta["train_balance_before_smote"]
    bal_after = meta["train_balance_after_smote"]
    c1, c2 = st.columns(2)
    c1.metric(
        "Training set before SMOTE",
        f"{bal_before['churned']:,} churned / {bal_before['stayed']:,} stayed",
        f"{bal_before['churn_rate']}% churn",
    )
    c2.metric(
        "Training set after SMOTE",
        f"{bal_after['churned']:,} churned / {bal_after['stayed']:,} stayed",
        f"{bal_after['churn_rate']}% churn",
    )

    results = pd.DataFrame(meta["results"]).T.reset_index(names="model")
    st.dataframe(results, use_container_width=True, hide_index=True)

    melted = results.melt(id_vars="model", var_name="metric", value_name="score")
    fig = px.bar(
        melted[melted["metric"].isin(["recall", "f1"])],
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Recall & F1 on churners — with vs without SMOTE",
    )
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        "SMOTE variants trade a little precision for a substantial recall gain on the "
        "churn class — exactly the customers a retention campaign must not miss."
    )

# ---------------------------------------------------------------- Gap 2
with tab_gap2:
    st.subheader("Gap 2 — Prediction without explanation")
    st.markdown(
        "Most reviewed papers stop at *who* will churn. Following the newest work "
        "(Asif et al., 2025; El Attar & El-Hajj, 2026), SHAP shows *why* — globally "
        "and for each individual customer."
    )

    with st.spinner("Computing SHAP values..."):
        imp = cached_global_importance(top_n=15)

    fig = px.bar(
        imp.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global drivers of churn (mean |SHAP value|)",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explain one customer")
    top_risky = risk_all.sort_values(ascending=False).head(50)
    options = {
        f"{df_raw.loc[i, 'customerID']} — risk {risk_all[i]:.0%}": i for i in top_risky.index
    }
    choice = st.selectbox("Pick a high-risk customer", list(options.keys()))
    idx = options[choice]
    contrib = explain_customer(explainer, X_all.loc[[idx]], top_n=10)
    contrib["direction"] = contrib["shap_value"].apply(
        lambda v: "pushes toward churn" if v > 0 else "pushes toward staying"
    )
    fig = px.bar(
        contrib.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        color="direction",
        orientation="h",
        color_discrete_map={
            "pushes toward churn": "#d62728",
            "pushes toward staying": "#2ca02c",
        },
        title=f"Why the model flags {df_raw.loc[idx, 'customerID']}",
        labels={"shap_value": "SHAP value", "feature": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Gap 3
with tab_gap3:
    st.subheader("Gap 3 — Nobody tests whether fixing the problem works")
    st.markdown(
        "The biggest gap across all 20 papers: predictions and explanations are delivered, "
        "but the retention action itself is never simulated. Here, realistic interventions "
        "are applied to the customer record and the model re-scores the risk — "
        "**before vs after the fix**."
    )

    st.markdown("#### Single customer")
    top_risky = risk_all.sort_values(ascending=False).head(50)
    options = {
        f"{df_raw.loc[i, 'customerID']} — risk {risk_all[i]:.0%}": i for i in top_risky.index
    }
    choice = st.selectbox("Pick an at-risk customer", list(options.keys()), key="whatif")
    idx = options[choice]

    with st.expander("Customer profile"):
        st.dataframe(df_raw.loc[[idx]].T.rename(columns={idx: "value"}), use_container_width=True)

    sim = simulate_customer(model, df_raw.loc[[idx]], meta["feature_columns"])
    best = sim.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Current risk", f"{best['risk_before']:.0%}")
    c2.metric(
        f"After: {best['intervention']}",
        f"{best['risk_after']:.0%}",
        f"-{best['risk_reduction']:.0%}",
        delta_color="inverse",
    )
    c3.metric("Interventions tested", len(sim))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Before", x=sim["intervention"], y=sim["risk_before"], marker_color="#d62728")
    )
    fig.add_trace(
        go.Bar(name="After", x=sim["intervention"], y=sim["risk_after"], marker_color="#2ca02c")
    )
    fig.update_layout(
        barmode="group",
        title="Predicted churn risk before vs after each intervention",
        yaxis_tickformat=".0%",
        xaxis_tickangle=-25,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Whole at-risk cohort")
    threshold = st.slider("Risk threshold defining 'at-risk'", 0.3, 0.8, 0.5, 0.05)
    with st.spinner("Simulating interventions across the cohort..."):
        cohort = cached_cohort_sim(threshold)
    if cohort.empty:
        st.warning("No customers above this risk threshold.")
    else:
        st.dataframe(cohort, use_container_width=True, hide_index=True)
        st.success(
            f"Best cohort-level intervention: **{cohort.iloc[0]['intervention']}** — average "
            f"risk drops from {cohort.iloc[0]['avg_risk_before']:.0%} to "
            f"{cohort.iloc[0]['avg_risk_after']:.0%}, moving "
            f"{cohort.iloc[0]['customers_moved_below_threshold']} customers below the threshold. "
            "This is the tested, practical action the literature stops short of."
        )

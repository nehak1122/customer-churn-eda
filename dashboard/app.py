"""Customer Churn — Overall Analytics Dashboard.

Full churn dashboard for the Telco dataset:
  - Executive overview (KPIs, churn split, predicted-risk distribution)
  - EDA insights (contract, payment, services, tenure, charges, demographics)
  - Risk segmentation (predicted-risk tiers, revenue at risk, top at-risk list)
  - Gap-analysis from the 20-paper literature review (2015-2026):
      Gap 1: class imbalance handling (SMOTE) - measured, not assumed
      Gap 2: explainability (SHAP) - global and per-customer
      Gap 3: what-if intervention simulation - does fixing the problem work?

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

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")


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
def cached_risk() -> pd.Series:
    return churn_probability(model, df_raw, meta["feature_columns"])


@st.cache_data
def cached_global_importance(top_n: int = 15) -> pd.DataFrame:
    sample = X_all.sample(min(800, len(X_all)), random_state=42)
    return global_importance(explainer, sample, top_n=top_n)


@st.cache_data
def cached_cohort_sim(threshold: float) -> pd.DataFrame:
    return simulate_cohort(model, df_raw, meta["feature_columns"], risk_threshold=threshold)


risk_all = cached_risk()
churned_mask = df_raw["Churn"] == "Yes"


def churn_rate_by(col: str) -> pd.DataFrame:
    """Churn % and customer count per category of a raw column."""
    grouped = (
        df_raw.assign(churned=churned_mask.astype(int))
        .groupby(col)
        .agg(churn_rate=("churned", "mean"), customers=("churned", "size"))
        .reset_index()
    )
    grouped["churn_rate"] *= 100
    return grouped


def rate_bar(data: pd.DataFrame, x: str, title: str, color: str = "#1f77b4"):
    fig = px.bar(
        data,
        x=x,
        y="churn_rate",
        title=title,
        labels={"churn_rate": "Churn rate (%)"},
        text=data["churn_rate"].round(1),
    )
    fig.update_traces(marker_color=color, textposition="outside")
    fig.update_layout(yaxis_range=[0, max(55, data["churn_rate"].max() * 1.2)])
    return fig


st.title("📉 Customer Churn — Analytics Dashboard")
st.caption(
    "Telco Customer Churn: executive overview, EDA insights, risk segmentation, and the "
    "gap-analysis from 20 research papers (2015–2026) — imbalance handling, SHAP "
    "explainability, and what-if intervention testing."
)

(
    tab_overview,
    tab_eda,
    tab_risk,
    tab_gap1,
    tab_gap2,
    tab_gap3,
) = st.tabs(
    [
        "📊 Executive Overview",
        "🔎 EDA Insights",
        "🎯 Risk Segmentation",
        "⚖️ Gap 1 · Imbalance (SMOTE)",
        "🔍 Gap 2 · Explainability (SHAP)",
        "🧪 Gap 3 · What-If Simulator",
    ]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    churn_rate = churned_mask.mean() * 100
    monthly_rev = df_raw["MonthlyCharges"].sum()
    lost_rev = df_raw.loc[churned_mask, "MonthlyCharges"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{len(df_raw):,}")
    c2.metric("Churn rate", f"{churn_rate:.1f}%")
    c3.metric("Monthly revenue", f"${monthly_rev:,.0f}")
    c4.metric("Revenue lost to churn", f"${lost_rev:,.0f}/mo")
    c5.metric("At-risk customers (p ≥ 0.5)", f"{int((risk_all >= 0.5).sum()):,}")

    left, mid, right = st.columns([1, 1.2, 1.2])
    with left:
        fig = px.pie(
            names=["Stayed", "Churned"],
            values=[int((~churned_mask).sum()), int(churned_mask.sum())],
            hole=0.55,
            title="Churn split",
            color_discrete_sequence=["#2ca02c", "#d62728"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with mid:
        fig = px.histogram(
            df_raw.assign(risk=risk_all),
            x="risk",
            nbins=40,
            title="Predicted churn-risk distribution",
            labels={"risk": "Predicted churn probability"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        avg = (
            df_raw.assign(churned=churned_mask)
            .groupby("churned")[["tenure", "MonthlyCharges"]]
            .mean()
            .rename(index={False: "Stayed", True: "Churned"})
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Avg tenure (months)", x=avg["churned"], y=avg["tenure"]))
        fig.add_trace(
            go.Bar(name="Avg monthly charges ($)", x=avg["churned"], y=avg["MonthlyCharges"])
        )
        fig.update_layout(barmode="group", title="Stayed vs churned — profile")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Model in production:** {meta['best_model']} "
        f"(F1 {meta['results'][meta['best_model']]['f1']:.2f}, "
        f"ROC-AUC {meta['results'][meta['best_model']]['roc_auc']:.2f} on held-out test data). "
        "See the Gap tabs for how it was selected, explained, and turned into actions."
    )

# ---------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("What drives churn in the data")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            rate_bar(churn_rate_by("Contract"), "Contract", "Churn rate by contract type"),
            use_container_width=True,
        )
    with r1c2:
        st.plotly_chart(
            rate_bar(
                churn_rate_by("PaymentMethod"), "PaymentMethod", "Churn rate by payment method"
            ),
            use_container_width=True,
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(
            rate_bar(
                churn_rate_by("InternetService"),
                "InternetService",
                "Churn rate by internet service",
            ),
            use_container_width=True,
        )
    with r2c2:
        bins = [0, 6, 12, 24, 48, 72]
        labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]
        tenure_grp = (
            df_raw.assign(
                churned=churned_mask.astype(int),
                tenure_group=pd.cut(df_raw["tenure"], bins=bins, labels=labels, right=True),
            )
            .groupby("tenure_group", observed=True)["churned"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
        )
        st.plotly_chart(
            rate_bar(tenure_grp, "tenure_group", "Churn rate by tenure (months)", "#ff7f0e"),
            use_container_width=True,
        )

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        services = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]
        rows = []
        for svc in services:
            sub = df_raw[df_raw[svc].isin(["Yes", "No"])]
            for has in ["Yes", "No"]:
                mask = sub[svc] == has
                rows.append(
                    {
                        "service": svc,
                        "has_service": "With" if has == "Yes" else "Without",
                        "churn_rate": (sub.loc[mask, "Churn"] == "Yes").mean() * 100,
                    }
                )
        fig = px.bar(
            pd.DataFrame(rows),
            x="service",
            y="churn_rate",
            color="has_service",
            barmode="group",
            title="Add-on services protect against churn",
            labels={"churn_rate": "Churn rate (%)", "has_service": ""},
            color_discrete_map={"With": "#2ca02c", "Without": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with r3c2:
        fig = px.histogram(
            df_raw.assign(Churn=df_raw["Churn"]),
            x="MonthlyCharges",
            color="Churn",
            nbins=40,
            barmode="overlay",
            opacity=0.6,
            title="Monthly charges distribution by churn",
            color_discrete_map={"No": "#2ca02c", "Yes": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Demographics")
    demo_rows = []
    for col, yes_label, no_label in [
        ("SeniorCitizen", "Senior citizen", "Not senior"),
        ("Partner", "Has partner", "No partner"),
        ("Dependents", "Has dependents", "No dependents"),
    ]:
        yes_mask = df_raw[col].isin([1, "Yes"])
        demo_rows.append(
            {"group": yes_label, "churn_rate": (df_raw.loc[yes_mask, "Churn"] == "Yes").mean() * 100}
        )
        demo_rows.append(
            {"group": no_label, "churn_rate": (df_raw.loc[~yes_mask, "Churn"] == "Yes").mean() * 100}
        )
    fig = px.bar(
        pd.DataFrame(demo_rows),
        x="group",
        y="churn_rate",
        title="Churn rate by demographic group",
        labels={"churn_rate": "Churn rate (%)", "group": ""},
        text=[f"{r['churn_rate']:.1f}" for r in demo_rows],
    )
    fig.update_traces(marker_color="#9467bd", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Headline EDA findings: month-to-month contracts (~43% churn vs ~3% on two-year), "
        "electronic-check payment, fiber-optic internet without add-on protections, and the "
        "first 6 months of tenure are the strongest churn signals — consistent with the "
        "notebook analysis in `Customer_Churn_EDA.ipynb`."
    )

# ---------------------------------------------------------------- Risk segmentation
with tab_risk:
    st.subheader("Predicted-risk segmentation")
    tiers = pd.cut(
        risk_all,
        bins=[-0.001, 0.3, 0.6, 1.0],
        labels=["Low (<30%)", "Medium (30–60%)", "High (>60%)"],
    )
    seg = df_raw.assign(risk=risk_all, tier=tiers)

    counts = seg["tier"].value_counts().reindex(["Low (<30%)", "Medium (30–60%)", "High (>60%)"])
    high_rev = seg.loc[seg["tier"] == "High (>60%)", "MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low risk", f"{int(counts.iloc[0]):,}")
    c2.metric("Medium risk", f"{int(counts.iloc[1]):,}")
    c3.metric("High risk", f"{int(counts.iloc[2]):,}")
    c4.metric("Monthly revenue in high tier", f"${high_rev:,.0f}")

    left, right = st.columns([1, 1.4])
    with left:
        fig = px.bar(
            counts.reset_index().set_axis(["tier", "customers"], axis=1),
            x="tier",
            y="customers",
            title="Customers per risk tier",
            color="tier",
            color_discrete_sequence=["#2ca02c", "#ff7f0e", "#d62728"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        tier_profile = (
            seg.groupby("tier", observed=True)
            .agg(
                avg_tenure=("tenure", "mean"),
                avg_monthly=("MonthlyCharges", "mean"),
                pct_month_to_month=("Contract", lambda s: (s == "Month-to-month").mean() * 100),
            )
            .round(1)
            .reset_index()
        )
        st.markdown("**Tier profiles**")
        st.dataframe(tier_profile, use_container_width=True, hide_index=True)
        st.caption(
            "High-risk customers skew to short tenure, high monthly charges and "
            "month-to-month contracts — the levers the What-If Simulator tests."
        )

    st.markdown("#### Top 20 at-risk customers")
    top = (
        seg.sort_values("risk", ascending=False)
        .head(20)[
            [
                "customerID",
                "risk",
                "tenure",
                "Contract",
                "PaymentMethod",
                "InternetService",
                "MonthlyCharges",
                "Churn",
            ]
        ]
        .assign(risk=lambda d: (d["risk"] * 100).round(1))
        .rename(columns={"risk": "risk %"})
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

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

# Gap Analysis — From 20 Research Papers (2015–2026) to Code

This project extends the original EDA with an implementation of the three research gaps
identified in our literature review of 20 verified customer churn papers published
between 2015 and 2026 (Vafeiadis et al. 2015 → El Attar & El-Hajj 2026).

## What the literature already does well

- **Model comparison**: ensembles and boosting (Random Forest, XGBoost, AdaBoost) consistently
  beat single classifiers on churn data (Vafeiadis et al., 2015; Ahmad et al., 2019; Lalwani et al., 2022).
- **Deep learning**: hybrid architectures such as BiLSTM-CNN (Khattak et al., 2023) and
  ChurnNet (Saha et al., 2024) push benchmark accuracy further.
- **Explainability is emerging**: SHAP/LIME appear in the most recent work
  (Asif et al., 2025; El Attar & El-Hajj, 2026).

## The three gaps and how this repo closes them

| # | Gap found in the literature | Evidence from the review | Implementation here |
|---|---|---|---|
| 1 | **Class imbalance is rarely handled carefully.** Only ~26.5% of Telco customers churn; models trained on raw data look accurate while missing most actual churners. Studies that do balance rarely show the before/after effect. | Amin et al. (2016) and Zhu et al. (2017) show the balancing technique matters as much as the classifier and that metric choice changes the winner; most applied papers (e.g. Jain et al., 2020) skip balancing entirely. | [`ml/train_model.py`](ml/train_model.py) trains Logistic Regression, Random Forest and XGBoost **with and without SMOTE** and evaluates all six variants on the same untouched test set, reporting accuracy, precision, recall, F1 and ROC-AUC so the recall gain on churners is measured, not assumed. |
| 2 | **Prediction without explanation.** Most papers stop at "who will churn" — business teams get a score but not the reasons in actionable terms. | Explainability only becomes standard in the 2025–2026 papers (Asif et al., 2025; El Attar & El-Hajj, 2026); everything earlier is mostly a black box or plain feature importance. | [`ml/shap_explainer.py`](ml/shap_explainer.py) provides **SHAP global importance** (which factors drive churn overall) and **per-customer explanations** (which factors push this specific customer toward leaving). |
| 3 | **Nobody tests whether fixing the problem works.** Across all 20 papers, prediction and explanation are delivered but the retention action itself is never simulated or evaluated — the loop from insight to intervention stays open. | Even the profit-driven stream (Stripling et al., 2018; Höppner et al., 2020) optimizes the model for profit but does not simulate individual retention actions; El Attar & El-Hajj (2026) quantify risk factors but stop short of what-if testing. | [`ml/intervention_simulator.py`](ml/intervention_simulator.py) implements a **what-if intervention simulator**: realistic retention levers (contract upgrade, adding tech support / online security, switching to auto-pay, monthly-charge discounts) are applied to the raw customer record, re-encoded and re-scored, reporting predicted churn risk **before vs after the fix** — per customer and averaged across the whole at-risk cohort. |

## How to reproduce

```bash
pip install -r requirements.txt
python -m ml.train_model          # trains all 6 model variants, saves best + metrics
streamlit run dashboard/app.py    # interactive dashboard (all three gaps, live)
```

The full literature review (20 summaries + references with links) lives in the
accompanying research documents; the 20 papers are listed with clickable DOI links in
`Customer_Churn_Research_Papers.xlsx` of the research pack.

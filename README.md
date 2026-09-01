# Customer Churn EDA & Feature Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Exploratory Data Analysis on Telco Customer Churn dataset to identify key features driving customer churn behavior — now extended with a **research gap-analysis implementation** (SMOTE, SHAP, what-if intervention simulation) and an interactive **Streamlit dashboard**.

**TechnoHacks Internship Program | Data Science | Medium Level Task**

---

## 🆕 Gap-Analysis Extension (from 20 research papers, 2015–2026)

We reviewed 20 verified customer churn papers (Vafeiadis et al. 2015 → El Attar & El-Hajj 2026)
and found three gaps, each now implemented in this repo — see [GAP_ANALYSIS.md](GAP_ANALYSIS.md)
for the paper-by-paper reasoning:

| Gap in the literature | Implementation |
|---|---|
| 1. Class imbalance rarely handled carefully | [`ml/train_model.py`](ml/train_model.py) — LR / Random Forest / XGBoost trained **with & without SMOTE**, all evaluated on the same untouched test set |
| 2. Prediction without explanation | [`ml/shap_explainer.py`](ml/shap_explainer.py) — SHAP global churn drivers + per-customer explanations |
| 3. Nobody tests whether fixing the problem works | [`ml/intervention_simulator.py`](ml/intervention_simulator.py) — what-if simulator: contract upgrades, add-on services, auto-pay, discounts → predicted risk **before vs after the fix**, per customer and cohort-wide |

### Run the pipeline & dashboard

```bash
pip install -r requirements.txt
python -m ml.train_model          # trains 6 model variants, saves best model + metrics
streamlit run dashboard/app.py    # interactive gap-analysis dashboard
```

Dashboard tabs: **Overview** (risk distribution) · **Gap 1** (SMOTE before/after metrics) ·
**Gap 2** (SHAP global + single-customer) · **Gap 3** (what-if intervention simulator with
cohort-level results).

### Deploy to Render

The repo ships a [`render.yaml`](render.yaml) blueprint (single free-tier web service,
pre-trained artifacts included — no training at build time):

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New +** → **Blueprint**
2. Connect/select this repository (`nehak1122/customer-churn-eda`)
3. Click **Apply** — Render builds from `requirements-render.txt` and serves the dashboard

With `autoDeploy: true`, every push to `main` redeploys automatically.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Total Customers | 7,043 |
| Churn Rate | 26.54% |
| Top Driver | Contract Type (40% impact) |
| High-Risk Segment | 52.7% churn rate |

### Top 5 Churn Drivers

1. **Contract Type** - Month-to-month: 42.7% vs Two-year: 2.6%
2. **Online Security** - Without: 42% vs With: 15%
3. **Tech Support** - Without: 42% vs With: 15%
4. **Tenure** - First 6 months: 53% churn rate
5. **Payment Method** - Electronic check: 45% vs Auto-pay: 15%

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nehak1122/customer-churn-eda.git
   cd customer-churn-eda
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Customer_Churn_EDA.ipynb
   ```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| matplotlib | ≥3.7.0 | Visualizations |
| seaborn | ≥0.12.0 | Statistical plots |
| scikit-learn | ≥1.3.0 | Feature importance (mutual_info_classif) |
| scipy | ≥1.10.0 | Chi-square tests |
| jupyter | ≥1.0.0 | Notebook environment |

### Quick Install

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

---

## Project Structure

```
customer-churn-eda/
├── Customer_Churn_EDA.ipynb    # Main analysis notebook
├── Customer_Churn_EDA.pdf      # PDF export for submission
├── Customer_Churn_EDA.html     # HTML export
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── PRD_Churn_EDA_Medium.docx   # Requirements document
├── GAP_ANALYSIS.md             # Research gaps from 20 papers → code mapping
├── ml/                         # Gap-analysis ML package
│   ├── data_preprocessing.py   # Load, clean, encode (reusable raw→features transform)
│   ├── train_model.py          # 6 model variants (LR/RF/XGB × with/without SMOTE)
│   ├── shap_explainer.py       # SHAP global + per-customer explanations
│   ├── intervention_simulator.py  # What-if retention interventions (before/after risk)
│   └── artifacts/              # Trained best model + metrics.json
├── dashboard/
│   └── app.py                  # Streamlit gap-analysis dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── *.png                       # Generated visualizations
    ├── dashboard_executive.png
    ├── executive_summary.png
    ├── risk_segmentation.png
    ├── tenure_analysis.png
    ├── service_impact.png
    ├── payment_contract.png
    ├── feature_importance.png
    └── correlation_matrix.png
```

---

## Usage

### Running the Notebook

```bash
jupyter notebook Customer_Churn_EDA.ipynb
```

Or use Google Colab:
1. Upload `Customer_Churn_EDA.ipynb` to Google Colab
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the session
3. Run all cells

### Running in VS Code

1. Install the Jupyter extension
2. Open `Customer_Churn_EDA.ipynb`
3. Select Python kernel
4. Run all cells

---

## Dataset

**Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Feature | Description |
|---------|-------------|
| customerID | Unique customer identifier |
| gender | Male/Female |
| SeniorCitizen | Whether senior citizen (0/1) |
| Partner/Dependents | Family status |
| tenure | Months with company |
| PhoneService/MultipleLines | Phone services |
| InternetService | DSL/Fiber optic/No |
| OnlineSecurity/Backup/Protection/TechSupport | Add-on services |
| StreamingTV/Movies | Streaming services |
| Contract | Month-to-month/One year/Two year |
| PaperlessBilling | Yes/No |
| PaymentMethod | Payment type |
| MonthlyCharges/TotalCharges | Billing amounts |
| **Churn** | Target variable (Yes/No) |

---

## Visualizations

The notebook generates 8 business-focused visualizations:

- **Executive Dashboard** - Overview of key metrics
- **Risk Segmentation** - Customer risk categories
- **Tenure Analysis** - Churn by customer age
- **Service Impact** - Effect of add-on services
- **Payment & Contract** - Billing analysis
- **Feature Importance** - Top churn drivers
- **Correlation Matrix** - Feature relationships
- **Executive Summary** - Recommendations

---

## License

MIT License - feel free to use for educational purposes.

---

## Acknowledgments

- Dataset: IBM Sample Data Sets via Kaggle
- Program: TechnoHacks Internship

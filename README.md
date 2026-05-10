# Customer Churn EDA & Feature Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Exploratory Data Analysis on Telco Customer Churn dataset to identify key features driving customer churn behavior.

**TechnoHacks Internship Program | Data Science | Medium Level Task**

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

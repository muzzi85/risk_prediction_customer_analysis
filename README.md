
# Risk Prediction Customer Analysis — Repo 4

## Overview

This repository contains an explainable machine learning framework for customer credit risk prediction and behavioral risk analysis.

The project focuses on:

- Probability of Default (PD) modeling
- Explainable AI (XAI)
- Feature engineering
- SHAP-based interpretability
- Threshold optimization
- False positive analysis
- Governance-oriented risk analytics
- Portfolio monitoring foundations
- Temporal forecasting preparation

---

<img width="1536" height="1024" alt="ChatGPT Image May 24, 2026, 11_52_30 PM" src="https://github.com/user-attachments/assets/1cb5e064-1ad2-443f-ba90-8a4c26268742" />


# Repository Architecture

```text
risk_prediction_customer_analysis/
│
├── api/                    # API services and model endpoints
├── dashboard/              # Dashboards and UI applications
├── data/                   # Raw and processed datasets . Can not be loaded into GitHub but can be found here https://www.kaggle.com/competitions/home-credit-default-risk/data
├── explainability/         # SHAP and interpretability utilities
├── features/               # Feature engineering logic
├── governance/             # Validation and governance checks TBA
├── models/                 # Trained models and pipelines
├── monitoring/             # Monitoring and drift utilities
├── notebooks/              # Research and experimentation notebooks
├── orchestration/          # Workflow orchestration TBA
├── tests/                  # Unit and integration tests
│
├── requirements.txt
└── README.md
```

---

# Notebook Guide

## 01_data_landscape_analysis.ipynb

Initial exploratory data analysis (EDA):

- dataset structure
- missing values
- target imbalance
- feature distributions
- correlation exploration

---

## 01_data_landscape_analysis_V2.ipynb

Enhanced EDA version with:

- improved visualizations
- deeper statistical analysis
- anomaly inspection
- business interpretation

---

## 02_benchmark_reference_model.ipynb

Baseline benchmarking notebook.

Implements:

- reference ML model
- initial train/test split
- ROC-AUC benchmarking
- baseline feature evaluation

---

## 02_feature_engineering.ipynb

Core feature engineering notebook.

Features include:

- financial ratios
- credit utilization metrics
- repayment behavior metrics
- borrowing acceleration signals
- debt relationships

---

## 02_feature_sanitization_v2.ipynb

Feature quality control and validation.

Includes:

- outlier detection
- invalid feature inspection
- leakage checks
- feature consistency validation
- sanitization pipelines

---

## 03_behavioral_feature_engineering_v2.ipynb

Advanced behavioral risk feature engineering.

Examples:

- repayment instability
- recent delay trend
- behavioral risk score
- customer deterioration indicators
- borrowing behavior patterns

---

## 03_credit_risk_modeling.ipynb

Main Probability of Default (PD) modeling notebook.

Implements:

- XGBoost training
- threshold optimization
- model evaluation
- precision/recall analysis
- ROC-AUC validation

---

## 04_explainability_analysis.ipynb

Explainability and forensic analytics.

Includes:

- SHAP waterfall plots
- SHAP dependence plots
- false positive analysis
- customer-level explanations
- feature impact interpretation

---

## 05_portfolio_monitoring.ipynb

Portfolio-level monitoring.

Focus:

- risk segmentation
- customer cohorts
- risk drift analysis
- monitoring KPIs
- portfolio distributions

---

## 05_portfolio_monitoring_draft.ipynb

Draft experimental notebook for future monitoring workflows.

---

## 06_event_traceability_and_forensic_analysis.ipynb

Risk event traceability framework.

Includes:

- forensic event tracing
- explainability auditing
- decision investigation
- customer event reconstruction

---

## 07_report_generation.ipynb

Automated reporting framework.

Outputs:

- model reports
- explainability summaries
- governance reporting
- monitoring exports

---

## 09_api_services.ipynb

API prototyping notebook.

Includes:

- model serving concepts
- API endpoints
- prediction interfaces
- deployment preparation

---

## home-credit-default-risk-v6.ipynb

Integrated experimentation notebook combining:

- feature engineering
- PD modeling
- explainability
- validation workflows

---

# Current Modeling Capabilities

## Probability of Default (PD)

Current system predicts:

```python
TARGET = 0 or 1
```

Where:

- 1 = risky/default customer
- 0 = safe customer

---

# Current Feature Categories

## Financial Features

- DEBT_TO_INCOME
- CREDIT_TO_ANNUITY_RATIO
- ACTIVE_DEBT_RATIO

## Behavioral Features

- LATE_PAYMENT_COUNT
- RECENT_DELAY_TREND
- REPAYMENT_INSTABILITY
- BORROWING_ACCELERATION_RATIO

## Stability Features

- YEARS_EMPLOYED
- EMPLOYMENT_TO_AGE_RATIO

## Composite Features

- BEHAVIORAL_RISK_SCORE

---

# Explainability Framework

The repository includes extensive explainability tooling using SHAP.

Implemented analyses:

- SHAP waterfall plots
- SHAP bar plots
- dependence plots
- local explanations
- false positive investigations
- high-confidence prediction analysis

---

# Current Limitations

Current version is primarily:

- static,
- snapshot-based,
- application-level.

Missing capabilities:

- temporal forecasting
- survival modeling
- sequential customer trajectories
- dynamic risk evolution

---

# Next Roadmap

## Repo 5 — Temporal Forecasting

Planned additions:

- rolling averages
- utilization slopes
- volatility metrics
- deterioration acceleration
- horizon-based forecasting

Target:

```text
Probability of default within next 3 months
```

---

## Future Repositories

| Repo | Objective |
|---|---|
| Repo 5 | Temporal Forecasting |
| Repo 6 | Exposure at Default (EAD) |
| Repo 7 | Loss Given Default (LGD) |
| Repo 8 | Risk-Based Pricing |
| Repo 9 | Portfolio Monitoring Platform |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/muzzi85/risk_prediction_customer_analysis.git
```

---

## Create Environment

```bash
python -m venv .venv
```

Activate:

### Linux / Mac

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Jupyter

---

# Long-Term Vision

Build an explainable financial AI platform capable of:

- default prediction,
- temporal forecasting,
- risk monitoring,
- expected loss estimation,
- explainable pricing,
- governance-compatible lending intelligence.

---

# Author

Mustafa Alhamdi

Financial AI & Risk Analytics

# Supply Chain Risk Dashboard

An end-to-end machine learning project predicting late delivery risk for supply chain orders, with a live interactive dashboard.

🔗 **Live Dashboard:** https://supply-chain-risk-dashboard-shaoubqxuz7ejv2bmpjtls.streamlit.app

## What this project does
- Predicts late delivery probability for each order using XGBoost (ROC-AUC 0.75)
- Explains individual predictions using SHAP
- Provides a live dashboard for risk overview, order lookup, regional analysis, and an operations action queue

## Contents
- `SCP_APLlogistic.ipynb` — full data cleaning, feature engineering, modeling, and evaluation notebook
- `Dashboard.py` — Streamlit dashboard application
- `Research_Paper.docx` — full technical write-up
- `Executive_Summary.docx` — stakeholder-facing summary
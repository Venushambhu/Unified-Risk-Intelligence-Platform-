💼 Unified Financial Risk Intelligence Platform
🧠 Overview
The Unified Financial Risk Intelligence Platform is an AI-powered system that combines Credit Risk, Fraud Risk, and Market Risk modeling into a single unified analytical framework.
It leverages machine learning, anomaly detection, and time-series forecasting to provide a holistic risk index that helps financial institutions assess exposure and make informed lending and investment decisions.

🧩 Project Highlights

🔍 Credit Risk Model: Predicts loan default probability using supervised ML models (Logistic Regression, Random Forest).
⚠️ Fraud Risk Model: Detects fraudulent or anomalous transactions using Isolation Forest and outlier scores.
📈 Market Risk Model: Forecasts market volatility using ARIMA and Prophet on NIFTY-50 time series data.
🧮 Unified Risk Index (URI): Weighted scoring metric combining results from all three models for overall financial exposure.
📊 Streamlit Dashboard: Interactive, real-time analytics with SHAP-based explainability for model transparency.

⚙️ Tech Stack
Programming: Python
Libraries: Pandas, NumPy, scikit-learn, Statsmodels, Prophet, Streamlit, Plotly, Seaborn, SHAP
Tools: Git, Jupyter, Anaconda
Datasets: credit risk (kaggle),  Fraud risk  (kaggle), Yahoo Finance Nifty-50 (Market)

🔬 Model Components
💳 Credit Risk Model

Predicts default probability based on loan amount, income, credit history, and home ownership.
Uses Logistic Regression with feature scaling and SMOTE balancing.
Metrics: ROC-AUC, Precision, Recall, F1-score

🕵️ Fraud Risk Model

Uses Isolation Forest to detect anomalous transactions.
Calculates a fraud risk score based on model decision function.
Metrics: Precision, Recall, Confusion Matrix

📊 Market Risk Model

Time-series forecasting with ARIMA and Prophet for volatility estimation.
Generates rolling volatility and daily Market Risk Scores.
Metrics: RMSE, MAE

🧮 Unified Risk Index
URI=0.4×CreditRisk+0.4×FraudRisk+0.2×MarketRisk

Provides an overall exposure rating for financial institutions based on weighted model outputs.

🚀 Setup & Installation
# Clone the repository

📊 Dashboard Features

Credit Risk prediction
Fraud transaction detection summary
Market risk trend analysis (volatility & risk score)

Unified Risk Index visualization

SHAP-based model explainability for transparency

🧠 Evaluation Summary
Model	Metrics	Result
Credit Risk	ROC-AUC, F1, Recall	≥ 85% accuracy
Fraud Detection	ROC-AUC, Recall	≥ 90% anomaly detection
Market Risk	RMSE, MAE	Stable volatility forecast
Unified Index	Cross-validation	Reliable composite scoring

🔮 Future Enhancements

Integrate LSTM models for deep learning market forecasts.
API deployment for real-time model inference.
Add authentication and cloud-based dashboard hosting.
Extend unified scoring to institutional portfolio risk assessment.

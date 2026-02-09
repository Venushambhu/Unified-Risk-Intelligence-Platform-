    
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------
# 🔧 App Setup
# -------------------------------------
st.set_page_config(page_title="Unified Financial Risk Intelligence Dashboard", layout="wide", page_icon="💼")

st.title("💼 Unified Financial Risk Intelligence Platform")
st.markdown("### Integrated Credit, Fraud, and Market Risk Monitoring System")

# -------------------------------------
# 📦 Data Loaders
# -------------------------------------
@st.cache_data
def load_credit_data():
    return pd.read_csv("final_credit_risk_dataset.csv")

@st.cache_data
def load_fraud_data():
    return pd.read_csv("final_fraud_risk_dataset.csv")

@st.cache_data
def load_market_data():
    return pd.read_csv("marketriskscore.csv")

@st.cache_resource
def load_credit_model():
    return joblib.load("credit_risk_model.pkl")

credit_df = load_credit_data()
fraud_df = load_fraud_data()
market_df = load_market_data()
model = load_credit_model()

# -------------------------------------
# 🧭 Navigation Tabs
# -------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💳 Credit Risk", "🕵️ Fraud Risk", "📈 Market Risk", "🧮 Unified Risk Index", "🤖 Explainability"]
)

# -------------------------------------------------------------------
# 💳 CREDIT RISK TAB
# -------------------------------------------------------------------
with tab1:
    st.header("💳 Credit Risk Analytics")

    # ---- Summary metrics ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(credit_df))
    col2.metric("Defaults", int(credit_df["loan_status"].sum()))
    col3.metric("Avg Credit Risk Score", f"{credit_df['credit_risk_score'].mean():.2f}")

    # ---- 1. Risk score distribution ----
    st.subheader("1️⃣ Risk Score Distribution")
    fig = px.histogram(credit_df, x="credit_risk_score", color="risk_category", nbins=40)
    st.plotly_chart(fig, use_container_width=True)

    # ---- 2. Risk by loan grade ----
    st.subheader("2️⃣ Risk by Loan Grade")
    fig = px.box(credit_df, x="loan_grade", y="credit_risk_score", color="loan_grade")
    st.plotly_chart(fig, use_container_width=True)

    # ---- 3. Risk by loan intent ----
    st.subheader("3️⃣ Average Risk by Loan Intent")
    avg_intent = credit_df.groupby("loan_intent")["credit_risk_score"].mean().reset_index()
    fig = px.bar(avg_intent, x="loan_intent", y="credit_risk_score", color="loan_intent")
    st.plotly_chart(fig, use_container_width=True)

    # ---- 5. Loan amount vs risk ----
    st.subheader("5️⃣ Loan Amount vs Credit Risk")
    fig = px.scatter(
        credit_df, x="loan_amnt", y="credit_risk_score",
        color="risk_category", size="loan_percent_income"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 6. Income vs loan % income ----
    st.subheader("6️⃣ Income vs Loan % of Income")
    fig = px.density_heatmap(
        credit_df, x="person_income", y="loan_percent_income",
        z="credit_risk_score", color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 7. Credit history vs risk ----
    st.subheader("7️⃣ Credit History Length vs Risk")
    fig = px.box(
        credit_df, x="cb_person_cred_hist_length", y="credit_risk_score",
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 8. Default count by home ownership ----
    st.subheader("8️⃣ Defaults by Home Ownership Type")
    default_by_home = credit_df.groupby("person_home_ownership")["loan_status"].sum().reset_index()
    fig = px.bar(default_by_home, x="person_home_ownership", y="loan_status", color="person_home_ownership")
    st.plotly_chart(fig, use_container_width=True)

    # ---- 9. Correlation heatmap ----
    st.subheader("9️⃣ Feature Correlation Heatmap")
    corr = credit_df.select_dtypes(include=['float64','int64']).corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, cmap="Blues", annot=False)
    st.pyplot(plt)

    # ---- 10. Feature importance (if available) ----
    st.subheader("🔟 Model Feature Importance (Credit Risk Model)")
    try:
        clf = model.named_steps["clf"]
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importances not available: {e}")



# -------------------------------------------------------------------
# 🕵️ FRAUD RISK TAB
# -------------------------------------------------------------------
with tab2:
    st.header("🕵️ Fraud Risk Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(fraud_df))
    col2.metric("Potential Frauds", sum(fraud_df["AnomalyLabel"] == "Potential Fraud"))
    col3.metric("Avg Fraud Risk Score", f"{fraud_df['Fraud_Risk_Score'].mean():.2f}")

    st.subheader("Fraud Risk Score Distribution")
    fig = px.histogram(fraud_df, x="Fraud_Risk_Score", nbins=40, color="AnomalyLabel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High Risk Transactions (Top 10)")
    top_frauds = fraud_df.sort_values(by="Fraud_Risk_Score", ascending=False).head(10)
    st.dataframe(top_frauds[["TransactionID", "AccountID", "Fraud_Risk_Score", "AnomalyLabel"]])

# -------------------------------------------------------------------
# 📈 MARKET RISK TAB
# -------------------------------------------------------------------
with tab3:
    st.header("📈 Market Risk Analysis (Volatility Forecast)")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", len(market_df))
    col2.metric("Current Market Volatility", f"{market_df['Volatility'].iloc[-1]:.4f}")
    col3.metric("Latest Market Risk Score", f"{market_df['Market_Risk_Score'].iloc[-1]:.2f}")

    # 1️⃣ Market Risk Score Trend
    st.subheader("1️⃣ Market Risk Score Trend")
    fig = px.line(
        market_df,
        x="Date",
        y="Market_Risk_Score",
        title="Market Risk Score Trend",
        color_discrete_sequence=["#FFA500"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Volatility Over Time
    st.subheader("2️⃣ Volatility Over Time")
    fig2 = px.line(
        market_df,
        x="Date",
        y="Volatility",
        title="Volatility (30-Day Rolling)",
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Returns vs Volatility Comparison
    if "Returns" in market_df.columns:
        st.subheader("3️⃣ Market Returns vs Volatility")
        fig3 = px.line(
            market_df,
            x="Date",
            y=["Returns", "Volatility"],
            labels={"value": "Metric Value", "variable": "Metric"},
            title="Returns and Volatility Comparison"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # 4️⃣ Smoothed Market Risk (Rolling Mean)
    st.subheader("4️⃣ Smoothed Market Risk Score (Rolling Mean)")
    market_df["RollingRisk"] = market_df["Market_Risk_Score"].rolling(30).mean()
    fig4 = px.line(
        market_df,
        x="Date",
        y="RollingRisk",
        title="30-Day Rolling Market Risk Score",
        color_discrete_sequence=["#ff7f0e"]
    )
    st.plotly_chart(fig4, use_container_width=True)

    # 5️⃣ Distribution of Volatility
    st.subheader("5️⃣ Distribution of Market Volatility")
    fig5 = px.histogram(
        market_df,
        x="Volatility",
        nbins=40,
        color_discrete_sequence=["#2ca02c"],
        title="Distribution of Market Volatility"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # 6️⃣ Correlation Heatmap (optional if multiple numeric columns)
    num_cols = market_df.select_dtypes(include=['float64', 'int64'])
    if num_cols.shape[1] > 2:
        st.subheader("6️⃣ Correlation Heatmap of Market Indicators")
        corr = num_cols.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, cmap="RdBu_r", center=0)
        st.pyplot(plt)

# -------------------------------------------------------------------
# 🧮 UNIFIED RISK INDEX TAB
# -------------------------------------------------------------------
with tab4:
    st.header("🧮 Unified Financial Risk Index")

    credit_mean = credit_df["credit_risk_score"].mean()
    fraud_mean = fraud_df["Fraud_Risk_Score"].mean()
    market_mean = market_df["Market_Risk_Score"].mean()

    unified_index = 0.4*credit_mean + 0.4*fraud_mean + 0.2*market_mean

    st.metric("Unified Financial Risk Index (URI)", f"{unified_index:.2f}")

    st.info(f"""
    Unified Risk Index = 0.4 × Credit Risk + 0.4 × Fraud Risk + 0.2 × Market Risk  
    = 0.4({credit_mean:.2f}) + 0.4({fraud_mean:.2f}) + 0.2({market_mean:.2f}) = **{unified_index:.2f}**
    """)

    # Compare contribution visually
    fig = px.pie(
        values=[credit_mean, fraud_mean, market_mean],
        names=["Credit Risk", "Fraud Risk", "Market Risk"],
        title="Contribution to Unified Risk Index",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 🤖 EXPLAINABILITY TAB
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 🤖 EXPLAINABILITY TAB (ALL RISK DOMAINS)
# -------------------------------------------------------------------
with tab5:
    st.header("🤖 Model Explainability & Interpretability")

    st.markdown("### 💳 Credit Risk – SHAP Feature Importance")
    st.info("Explains how each borrower feature contributes to the credit risk prediction.")

    try:
        clf = model.named_steps["clf"]
        preproc = model.named_steps["preprocess"]
        X = credit_df.drop(columns=["credit_risk_score", "risk_category", "loan_status"])
        X_transformed = preproc.transform(X)

        # SHAP explainer for credit model
        explainer = shap.Explainer(clf, X_transformed)
        shap_values = explainer(X_transformed)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, X_transformed, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Credit Risk SHAP could not be displayed: {e}")

    # ----------------------------------------------------------------
    st.markdown("### 🕵️ Fraud Risk – Feature Impact Visualization")
    st.info("Highlights which transaction features most influence fraud detection (Isolation Forest).")

    try:
        features = ["TransactionAmount", "TransactionDuration", "AccountBalance", "LoginAttempts"]
        X_fraud = fraud_df[features]
        X_scaled = (X_fraud - X_fraud.mean()) / X_fraud.std()

        iso_explainer = shap.Explainer(iso_forest, X_scaled)
        iso_values = iso_explainer(X_scaled)

        fig2 = plt.figure(figsize=(10, 6))
        shap.summary_plot(iso_values.values, X_scaled, feature_names=features, show=False)
        st.pyplot(fig2)

        st.subheader("Top 10 Transactions by Fraud Risk")
        top_fraud = fraud_df.sort_values(by="Fraud_Risk_Score", ascending=False).head(10)
        st.dataframe(top_fraud[["TransactionID", "AccountID", "Fraud_Risk_Score", "AnomalyLabel"]])
    except Exception as e:
        st.warning(f"Fraud Risk explainability unavailable: {e}")

    # ----------------------------------------------------------------
    st.markdown("### 📈 Market Risk – Trend & Component Analysis")
    st.info("Visualizes trend and volatility components from the ARIMA/Prophet forecast model.")

    try:
        # Plot main market risk and volatility trends
        fig3 = px.line(market_df, x="Date", y="Market_Risk_Score", title="Market Risk Score Trend")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.line(market_df, x="Date", y="Volatility", title="Market Volatility Over Time")
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.warning(f"Market Risk trend visualization failed: {e}")

st.success("✅ Explainability & Interpretability Section Loaded Successfully!")


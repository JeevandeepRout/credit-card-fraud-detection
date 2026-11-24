"""
Credit Card Fraud Detection System — Home Page
================================================
Streamlit multi-page application entry point.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---------------------------------------------------------------------------
# Home content
# ---------------------------------------------------------------------------
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("---")

st.markdown(
    """
    Welcome to the **Credit Card Fraud Detection** dashboard — an academic
    demonstration that trains, evaluates, and compares machine-learning
    models on a highly imbalanced transaction dataset.

    ### 🚀 Quick-Start Guide

    | Step | Page | What happens |
    |------|------|--------------|
    | 1 | **📊 Data Exploration** | Load the dataset, inspect distributions, and visualise class imbalance. |
    | 2 | **🏋️ Model Training** | Train Logistic Regression, Random Forest, and XGBoost models. |
    | 3 | **📈 Model Evaluation** | Compare precision, recall, F1, PR-AUC, and confusion matrices. |
    | 4 | **🔍 Predict Transaction** | Submit a single transaction and get a fraud/legitimate prediction. |
    | 5 | **📋 Prediction History** | Review every prediction made during this session. |

    ### 📂 Dataset

    This application expects the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    (`creditcard.csv`) to be placed in the **`data/`** directory at the
    project root.

    The dataset contains **284 807 transactions** with 30 numeric features
    (`Time`, `V1`–`V28`, `Amount`) and a binary target column (`Class`).
    Features `V1`–`V28` are PCA-transformed and anonymised for privacy.
    """
)

# Quick stats if data has been loaded
if "dataset" in st.session_state:
    st.markdown("---")
    df = st.session_state.dataset
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Features", len(df.columns) - 1)
    n_fraud = int((df["Class"] == 1).sum())
    c3.metric("Fraud Cases", f"{n_fraud:,}")
    c4.metric("Fraud Rate", f"{n_fraud / len(df) * 100:.3f}%")

st.markdown("---")

st.info(
    "⚠️ **Disclaimer:** This is an academic project. Scores displayed are "
    "model outputs on the training dataset's test split and should NOT be "
    "interpreted as real-world fraud probabilities. Never use this system "
    "to make actual financial decisions.",
    icon="ℹ️",
)

st.sidebar.success("Select a page above to get started.")

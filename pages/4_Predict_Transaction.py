"""Page 4 — Interactive single-transaction prediction."""

from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import get_available_models, load_model
from src.prediction import predict_transaction, validate_input
from src.preprocessing import load_feature_names, load_scaler
from src.utils import ALL_FEATURE_COLUMNS, DEFAULT_THRESHOLD, TARGET_COLUMN

st.set_page_config(page_title="Predict Transaction", page_icon="🔍", layout="wide")
st.title("🔍 Predict Transaction")
st.markdown("---")

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
available = get_available_models()
if not available:
    st.warning(
        "No trained model artifacts found. "
        "Go to **🏋️ Model Training** to train and save models first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
model_name = c1.selectbox("Model", available)
threshold = c2.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.05,
)

# ---------------------------------------------------------------------------
# Load a sample transaction from the dataset
# ---------------------------------------------------------------------------
st.subheader("Input Features")

sample_values: dict[str, float] = {}

if "dataset" in st.session_state:
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("🎲 Load Random Legitimate"):
            df = st.session_state.dataset
            legit = df[df[TARGET_COLUMN] == 0]
            row = legit.sample(1, random_state=None).iloc[0]
            sample_values = {f: float(row[f]) for f in ALL_FEATURE_COLUMNS}
            st.session_state._sample_values = sample_values
    with col_btn2:
        if st.button("🎲 Load Random Fraud"):
            df = st.session_state.dataset
            fraud = df[df[TARGET_COLUMN] == 1]
            row = fraud.sample(1, random_state=None).iloc[0]
            sample_values = {f: float(row[f]) for f in ALL_FEATURE_COLUMNS}
            st.session_state._sample_values = sample_values

if "_sample_values" in st.session_state and not sample_values:
    sample_values = st.session_state._sample_values

# ---------------------------------------------------------------------------
# Feature input form
# ---------------------------------------------------------------------------
st.markdown(
    "Enter all **30 features** below. Use the *Load Random* buttons above "
    "to populate fields from the dataset."
)

feature_values: dict[str, float | str] = {}

with st.form("prediction_form"):
    # Time
    feature_values["Time"] = st.number_input(
        "Time (seconds since first transaction)",
        value=sample_values.get("Time", 0.0),
        format="%.4f",
    )

    # V1 – V28 in a grid
    st.markdown("**PCA Features (V1 – V28)**")
    n_cols = 4
    rows_needed = 7  # 28 / 4 = 7
    for row_i in range(rows_needed):
        cols = st.columns(n_cols)
        for col_i, col_widget in enumerate(cols):
            feat_idx = row_i * n_cols + col_i + 1
            if feat_idx > 28:
                break
            feat_name = f"V{feat_idx}"
            feature_values[feat_name] = col_widget.number_input(
                feat_name,
                value=sample_values.get(feat_name, 0.0),
                format="%.6f",
                key=f"input_{feat_name}",
            )

    # Amount
    feature_values["Amount"] = st.number_input(
        "Amount ($)",
        value=sample_values.get("Amount", 0.0),
        min_value=0.0,
        format="%.2f",
    )

    submitted = st.form_submit_button("🔎 Predict", type="primary")

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if submitted:
    try:
        result = predict_transaction(
            model_name=model_name,
            features=feature_values,
            threshold=threshold,
        )

        # Store in session history
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append(result)

        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")

        if result.predicted_class == "Fraud":
            st.error(f"🚨 **FRAUD** detected by {result.model_name}")
        else:
            st.success(f"✅ Transaction classified as **LEGITIMATE** by {result.model_name}")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Prediction", result.predicted_class)
        if result.score is not None:
            r2.metric("Model Score", f"{result.score:.6f}")
        else:
            r2.metric("Model Score", "N/A")
        r3.metric("Threshold", f"{result.threshold_used:.2f}")
        r4.metric("Prediction ID", result.prediction_id)

        st.caption(f"Timestamp: {result.timestamp}")

        # Disclaimer
        st.info(
            "⚠️ The **Model Score** is the model's output score, NOT a "
            "calibrated probability of real-world fraud. The prediction "
            "reflects the learned patterns from the training dataset only.",
            icon="ℹ️",
        )

    except ValueError as exc:
        st.error(f"Validation Error:\n\n{exc}")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

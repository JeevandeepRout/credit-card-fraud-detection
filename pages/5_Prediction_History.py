"""Page 5 — Session-based prediction history."""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import PredictionResult

st.set_page_config(page_title="Prediction History", page_icon="📋", layout="wide")
st.title("📋 Prediction History")
st.markdown("---")

# ---------------------------------------------------------------------------
# Check history
# ---------------------------------------------------------------------------
history: list[PredictionResult] = st.session_state.get("prediction_history", [])

if not history:
    st.info(
        "No predictions yet. Go to **🔍 Predict Transaction** to evaluate "
        "a transaction."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
n_total = len(history)
n_fraud = sum(1 for r in history if r.predicted_class == "Fraud")
n_legit = n_total - n_fraud

m1, m2, m3 = st.columns(3)
m1.metric("Total Predictions", n_total)
m2.metric("Flagged as Fraud", n_fraud)
m3.metric("Classified Legitimate", n_legit)

# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------
st.subheader("All Predictions (most recent first)")

rows = []
for r in reversed(history):
    rows.append(
        {
            "ID": r.prediction_id,
            "Timestamp": r.timestamp,
            "Model": r.model_name,
            "Prediction": r.predicted_class,
            "Score": f"{r.score:.6f}" if r.score is not None else "N/A",
            "Threshold": f"{r.threshold_used:.2f}",
        }
    )

df_hist = pd.DataFrame(rows)

# Colour-code the Prediction column
def _highlight_pred(val: str) -> str:
    if val == "Fraud":
        return "background-color: #f8d7da; color: #721c24;"
    return "background-color: #d4edda; color: #155724;"

# Support both pandas >= 2.1 (style.map) and legacy versions (style.applymap)
if hasattr(df_hist.style, "map"):
    styled = df_hist.style.map(_highlight_pred, subset=["Prediction"])
else:
    styled = df_hist.style.applymap(_highlight_pred, subset=["Prediction"])
st.dataframe(styled, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Clear history
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("🗑️ Clear History"):
    st.session_state.prediction_history = []
    st.rerun()

st.caption(
    "Prediction history is session-based and will be cleared when the "
    "application is restarted. No sensitive data is stored."
)

"""Page 3 — Evaluate and compare trained models."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    evaluate_model,
    evaluations_to_dataframe,
    plot_all_pr_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curve,
)
from src.utils import DEFAULT_THRESHOLD

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")
st.title("📈 Model Evaluation")
st.markdown("---")

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if "trained_models" not in st.session_state:
    st.warning("No trained models found. Go to **🏋️ Model Training** first.")
    st.stop()

if "X_test" not in st.session_state or "y_test" not in st.session_state:
    st.warning("Test data not found in session. Please re-train.")
    st.stop()

models = st.session_state.trained_models
X_test = st.session_state.X_test
y_test = st.session_state.y_test

# ---------------------------------------------------------------------------
# Threshold control
# ---------------------------------------------------------------------------
threshold = st.slider(
    "Classification threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="Predictions with model score ≥ this threshold are classified as Fraud.",
)

# ---------------------------------------------------------------------------
# Evaluate all models
# ---------------------------------------------------------------------------
evaluations = []
for name, model in models.items():
    ev = evaluate_model(model, X_test, y_test, name, threshold=threshold)
    evaluations.append(ev)

st.session_state.evaluations = evaluations

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
st.subheader("Performance Comparison")
comp_df = evaluations_to_dataframe(evaluations)
st.dataframe(
    comp_df.style.highlight_max(
        subset=["Precision", "Recall", "F1-Score", "PR-AUC"],
        color="#d4edda",
    ).highlight_min(
        subset=["FPR"],
        color="#d4edda",
    ),
    width="stretch",
    hide_index=True,
)

# Flag small test sample
n_fraud_test = int(y_test.sum())
if n_fraud_test < 30:
    st.warning(
        f"⚠️ Only **{n_fraud_test}** fraud cases in the test set. "
        "Metrics may not be reliable."
    )

# ---------------------------------------------------------------------------
# Bar chart comparison
# ---------------------------------------------------------------------------
st.subheader("Visual Comparison")
fig_comp = plot_model_comparison(evaluations)
st.pyplot(fig_comp)

# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------
st.subheader("Precision-Recall Curves")
fig_pr = plot_all_pr_curves(models, X_test, y_test)
st.pyplot(fig_pr)

# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
st.subheader("Confusion Matrices")
cm_cols = st.columns(len(evaluations))
for col, ev in zip(cm_cols, evaluations):
    with col:
        fig_cm = plot_confusion_matrix(ev)
        st.pyplot(fig_cm)

# ---------------------------------------------------------------------------
# Per-model detail
# ---------------------------------------------------------------------------
st.subheader("Individual Model Details")
selected = st.selectbox("Select a model for details", [ev.model_name for ev in evaluations])

sel_ev = next(ev for ev in evaluations if ev.model_name == selected)
sel_model = models[selected]

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Precision", f"{sel_ev.precision:.4f}")
d2.metric("Recall", f"{sel_ev.recall:.4f}")
d3.metric("F1-Score", f"{sel_ev.f1_score:.4f}")
d4.metric("PR-AUC", f"{sel_ev.pr_auc:.4f}")
d5.metric("FPR", f"{sel_ev.false_positive_rate:.6f}")

fig_pr_single = plot_precision_recall_curve(sel_model, X_test, y_test, selected)
st.pyplot(fig_pr_single)

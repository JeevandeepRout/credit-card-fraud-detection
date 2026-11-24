"""Page 2 — Train Logistic Regression, Random Forest, and XGBoost."""

from __future__ import annotations

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import save_model, train_model
from src.preprocessing import preprocess_and_split, save_feature_names, save_scaler
from src.utils import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, SUPPORTED_MODELS

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")
st.title("🏋️ Model Training")
st.markdown("---")

# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------
if "dataset" not in st.session_state:
    st.warning("No dataset loaded. Go to **📊 Data Exploration** first.")
    st.stop()

df = st.session_state.dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.subheader("Training Configuration")

c1, c2 = st.columns(2)
test_size = c1.slider("Test size", 0.1, 0.4, DEFAULT_TEST_SIZE, 0.05)
random_state = c2.number_input("Random state", value=DEFAULT_RANDOM_STATE, step=1)

models_to_train = st.multiselect(
    "Models to train",
    list(SUPPORTED_MODELS.keys()),
    default=list(SUPPORTED_MODELS.keys()),
)

if not models_to_train:
    st.info("Select at least one model to train.")
    st.stop()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if st.button("🚀 Train Selected Models", type="primary"):
    # 1. Preprocess & split
    with st.spinner("Preprocessing and splitting data …"):
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(
            df, test_size=test_size, random_state=random_state,
        )
        save_scaler(scaler)
        save_feature_names(feature_names)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_names

    st.success(
        f"Data split: **{len(X_train):,}** train / **{len(X_test):,}** test  "
        f"(fraud in test: **{int(y_test.sum()):,}**)"
    )

    # 2. Train each model
    trained: dict = {}
    progress = st.progress(0, text="Starting training …")

    for idx, name in enumerate(models_to_train):
        progress.progress(
            (idx) / len(models_to_train),
            text=f"Training **{name}** …",
        )
        t0 = time.perf_counter()
        try:
            model = train_model(name, X_train, y_train, random_state=random_state)
            elapsed = time.perf_counter() - t0
            save_model(model, name)
            trained[name] = model
            st.success(f"✅ **{name}** trained in {elapsed:.1f}s and saved to `artifacts/`.")
        except Exception as exc:
            st.error(f"❌ **{name}** failed: {exc}")

    progress.progress(1.0, text="All done!")

    if trained:
        st.session_state.trained_models = trained
        st.balloons()

# ---------------------------------------------------------------------------
# Show current state
# ---------------------------------------------------------------------------
if "trained_models" in st.session_state:
    st.markdown("---")
    st.subheader("Trained Models in Session")
    for name in st.session_state.trained_models:
        st.markdown(f"- ✅ **{name}**")
    st.info("Proceed to **📈 Model Evaluation** to compare these models.")

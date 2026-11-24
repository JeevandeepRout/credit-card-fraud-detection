"""Page 1 — Exploratory Data Analysis."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import get_class_distribution, get_summary, load_dataset, validate_schema
from src.utils import ALL_FEATURE_COLUMNS, DEFAULT_DATASET_PATH, TARGET_COLUMN

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")
st.title("📊 Data Exploration")
st.markdown("---")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset_path = st.text_input(
    "Path to `creditcard.csv`",
    value=DEFAULT_DATASET_PATH,
    help="Absolute or relative path to the Kaggle credit-card fraud CSV.",
)

if st.button("Load Dataset", type="primary") or "dataset" in st.session_state:
    try:
        if "dataset" not in st.session_state or st.session_state.get("_dataset_path") != dataset_path:
            with st.spinner("Loading dataset …"):
                df = load_dataset(dataset_path)
            st.session_state.dataset = df
            st.session_state._dataset_path = dataset_path
        df = st.session_state.dataset

        # Schema check
        is_valid, schema_errors = validate_schema(df)
        if not is_valid:
            st.error("Schema validation failed:")
            for err in schema_errors:
                st.warning(err)
            st.stop()
        elif schema_errors:
            for err in schema_errors:
                st.info(err)

        st.success(f"Dataset loaded — **{len(df):,}** rows × **{len(df.columns)}** columns")

        # ----- Summary statistics -----
        st.subheader("Summary Statistics")
        summary = get_summary(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{summary['n_rows']:,}")
        m2.metric("Columns", summary["n_columns"])
        m3.metric("Missing Values", summary["total_missing"])

        with st.expander("Descriptive Statistics", expanded=False):
            st.dataframe(summary["describe"], width="stretch")

        # ----- Class distribution -----
        st.subheader("Class Distribution")
        dist = get_class_distribution(df)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Legitimate", f"{dist['legitimate']:,}")
        d2.metric("Fraud", f"{dist['fraud']:,}")
        d3.metric("Fraud %", f"{dist['fraud_percentage']:.4f}%")
        d4.metric("Imbalance Ratio", f"{dist['imbalance_ratio']:.0f}:1")

        # Bar chart
        fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
        bars = ax_dist.bar(
            ["Legitimate (0)", "Fraud (1)"],
            [dist["legitimate"], dist["fraud"]],
            color=["#3498db", "#e74c3c"],
        )
        ax_dist.bar_label(bars, fmt="%d", fontsize=9)
        ax_dist.set_ylabel("Count")
        ax_dist.set_title("Transaction Class Distribution")
        fig_dist.tight_layout()
        st.pyplot(fig_dist)

        # ----- Feature distributions -----
        st.subheader("Feature Distributions")

        col_option = st.selectbox(
            "Select a feature to explore",
            ALL_FEATURE_COLUMNS,
            index=ALL_FEATURE_COLUMNS.index("Amount"),
        )

        fig_feat, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram by class
        for cls, label, color in [(0, "Legitimate", "#3498db"), (1, "Fraud", "#e74c3c")]:
            subset = df[df[TARGET_COLUMN] == cls][col_option]
            axes[0].hist(subset, bins=80, alpha=0.6, label=label, color=color, density=True)
        axes[0].set_title(f"{col_option} — Distribution by Class")
        axes[0].set_xlabel(col_option)
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # Box plot by class
        data_box = [
            df[df[TARGET_COLUMN] == 0][col_option].values,
            df[df[TARGET_COLUMN] == 1][col_option].values,
        ]
        bp = axes[1].boxplot(data_box, labels=["Legitimate", "Fraud"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[1].set_title(f"{col_option} — Box Plot by Class")
        axes[1].set_ylabel(col_option)

        fig_feat.tight_layout()
        st.pyplot(fig_feat)

        # ----- Correlation heatmap (top features) -----
        st.subheader("Correlation with Target")
        corr = df[ALL_FEATURE_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
        top_features = corr.abs().sort_values(ascending=False).head(15)

        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v < 0 else "#3498db" for v in corr[top_features.index]]
        ax_corr.barh(top_features.index[::-1], corr[top_features.index][::-1], color=colors[::-1])
        ax_corr.set_xlabel("Pearson Correlation with Class")
        ax_corr.set_title("Top 15 Features Correlated with Fraud")
        ax_corr.axvline(0, color="grey", linewidth=0.8)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)

        # ----- Time & Amount plots -----
        st.subheader("Time & Amount Analysis")
        t1, t2 = st.columns(2)

        with t1:
            fig_time, ax_time = plt.subplots(figsize=(6, 3.5))
            ax_time.hist(df[df[TARGET_COLUMN] == 0]["Time"], bins=100, alpha=0.5, label="Legitimate", color="#3498db", density=True)
            ax_time.hist(df[df[TARGET_COLUMN] == 1]["Time"], bins=100, alpha=0.7, label="Fraud", color="#e74c3c", density=True)
            ax_time.set_xlabel("Time (seconds)")
            ax_time.set_ylabel("Density")
            ax_time.set_title("Transaction Time Distribution")
            ax_time.legend()
            fig_time.tight_layout()
            st.pyplot(fig_time)

        with t2:
            fig_amt, ax_amt = plt.subplots(figsize=(6, 3.5))
            ax_amt.hist(df[df[TARGET_COLUMN] == 0]["Amount"], bins=100, alpha=0.5, label="Legitimate", color="#3498db", density=True, range=(0, 500))
            ax_amt.hist(df[df[TARGET_COLUMN] == 1]["Amount"], bins=100, alpha=0.7, label="Fraud", color="#e74c3c", density=True, range=(0, 500))
            ax_amt.set_xlabel("Amount ($)")
            ax_amt.set_ylabel("Density")
            ax_amt.set_title("Transaction Amount Distribution (up to $500)")
            ax_amt.legend()
            fig_amt.tight_layout()
            st.pyplot(fig_amt)

        # ----- Raw data preview -----
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100), width="stretch")

    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

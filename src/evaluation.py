"""Model evaluation metrics and comparison visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from .utils import ModelEvaluation


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> ModelEvaluation:
    """Evaluate *model* on the held-out test set and return metrics."""
    has_proba = hasattr(model, "predict_proba")

    if has_proba:
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_scores = None

    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    pr_auc = 0.0
    if y_scores is not None and len(np.unique(y_test)) > 1:
        pr_auc = average_precision_score(y_test, y_scores)

    return ModelEvaluation(
        model_name=model_name,
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1_score=f1_score(y_test, y_pred, zero_division=0),
        pr_auc=pr_auc,
        confusion_matrix=cm.tolist(),
        false_positive_rate=fpr,
    )


# ---------------------------------------------------------------------------
# Tabular comparison
# ---------------------------------------------------------------------------
def evaluations_to_dataframe(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for ev in evaluations:
        rows.append(
            {
                "Model": ev.model_name,
                "Precision": round(ev.precision, 4),
                "Recall": round(ev.recall, 4),
                "F1-Score": round(ev.f1_score, 4),
                "PR-AUC": round(ev.pr_auc, 4),
                "FPR": round(ev.false_positive_rate, 6),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(evaluation: ModelEvaluation) -> plt.Figure:
    cm = np.array(evaluation.confusion_matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {evaluation.model_name}")
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_scores)
        ap = average_precision_score(y_test, y_scores)
        ax.plot(rec, prec, lw=2, label=f"{model_name} (AP = {ap:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5,
            "Probability scores not available for this model.",
            ha="center", va="center", transform=ax.transAxes,
        )

    ax.set_title(f"Precision-Recall Curve — {model_name}")
    fig.tight_layout()
    return fig


def plot_all_pr_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """Overlay PR curves for every model in *models* (name → fitted model)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_scores = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_scores)
        ap = average_precision_score(y_test, y_scores)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_model_comparison(evaluations: list[ModelEvaluation]) -> plt.Figure:
    """Grouped bar chart comparing models across key metrics."""
    df = evaluations_to_dataframe(evaluations)
    metrics = ["Precision", "Recall", "F1-Score", "PR-AUC"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    n = len(df)
    width = 0.8 / max(n, 1)

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=row["Model"])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

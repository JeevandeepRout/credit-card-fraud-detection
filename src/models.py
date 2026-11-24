"""Model training, saving, and loading."""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .utils import ARTIFACTS_DIR, SUPPORTED_MODELS, ensure_dir


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Logistic Regression with class-weight balancing."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Random Forest with class-weight balancing."""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> XGBClassifier:
    """XGBoost with *scale_pos_weight* for class imbalance."""
    n_legit = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    spw = n_legit / n_fraud if n_fraud > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=spw,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# Dispatch table --------------------------------------------------------
_TRAIN_FN = {
    "Logistic Regression": train_logistic_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
}


def train_model(name: str, X_train, y_train, random_state: int = 42):
    """Train a model by its display name."""
    fn = _TRAIN_FN.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(_TRAIN_FN)}"
        )
    return fn(X_train, y_train, random_state)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _artifact_path(model_name: str) -> str:
    slug = SUPPORTED_MODELS.get(model_name, model_name.lower().replace(" ", "_"))
    return os.path.join(ARTIFACTS_DIR, f"{slug}.joblib")


def save_model(model, model_name: str, path: str | None = None) -> str:
    ensure_dir(ARTIFACTS_DIR)
    path = path or _artifact_path(model_name)
    joblib.dump(model, path)
    return path


def load_model(model_name: str, path: str | None = None):
    path = path or _artifact_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at '{path}'. Train it first."
        )
    return joblib.load(path)


def get_available_models() -> list[str]:
    """Return display names of models whose artifacts exist on disk."""
    return [
        name
        for name in SUPPORTED_MODELS
        if os.path.exists(_artifact_path(name))
    ]

"""Feature preprocessing, train/test splitting, and scaler management."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import (
    ALL_FEATURE_COLUMNS,
    ARTIFACTS_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    SCALE_FEATURES,
    TARGET_COLUMN,
    ensure_dir,
)


# ---------------------------------------------------------------------------
# Split + fit scaler
# ---------------------------------------------------------------------------
def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, list[str]]:
    """Split data with stratification and fit the scaler on **training data only**.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    cols_to_scale = [c for c in SCALE_FEATURES if c in X_train.columns]

    if cols_to_scale:
        X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# Scaler persistence
# ---------------------------------------------------------------------------
def save_scaler(scaler: StandardScaler, path: str | None = None) -> str:
    if path is None:
        ensure_dir(ARTIFACTS_DIR)
        path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    joblib.dump(scaler, path)
    return path


def load_scaler(path: str | None = None) -> StandardScaler:
    if path is None:
        path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at '{path}'. Train models first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Feature-name persistence
# ---------------------------------------------------------------------------
def save_feature_names(feature_names: list[str], path: str | None = None) -> str:
    if path is None:
        ensure_dir(ARTIFACTS_DIR)
        path = os.path.join(ARTIFACTS_DIR, "feature_names.joblib")
    joblib.dump(feature_names, path)
    return path


def load_feature_names(path: str | None = None) -> list[str]:
    if path is None:
        path = os.path.join(ARTIFACTS_DIR, "feature_names.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature names not found at '{path}'. Train models first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Single-row preprocessing (inference)
# ---------------------------------------------------------------------------
def preprocess_input(
    features: dict[str, Any],
    scaler: StandardScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply the **same** preprocessing used during training to one row.

    Raises ValueError if required features are missing.
    """
    missing = set(feature_names) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    input_df = pd.DataFrame([features])[feature_names]

    cols_to_scale = [c for c in SCALE_FEATURES if c in input_df.columns]
    if cols_to_scale:
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    return input_df

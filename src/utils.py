"""Shared constants, data classes, and utility functions."""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# The Kaggle credit-card fraud dataset columns:
#   Time, V1–V28, Amount, Class
ANONYMIZED_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
ALL_FEATURE_COLUMNS: list[str] = ["Time"] + ANONYMIZED_FEATURES + ["Amount"]
TARGET_COLUMN: str = "Class"
EXPECTED_COLUMNS: list[str] = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]

# Only Time and Amount need scaling; V1–V28 are already PCA-transformed.
SCALE_FEATURES: list[str] = ["Time", "Amount"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR: str = os.path.join(PROJECT_DIR, "artifacts")
DATA_DIR: str = os.path.join(PROJECT_DIR, "data")
DEFAULT_DATASET_PATH: str = os.path.join(DATA_DIR, "creditcard.csv")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, str] = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

DEFAULT_THRESHOLD: float = 0.5
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Result of a single transaction prediction."""

    prediction_id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    model_name: str = ""
    predicted_class: str = ""          # "Fraud" or "Legitimate"
    score: Optional[float] = None      # model probability / raw score
    threshold_used: float = DEFAULT_THRESHOLD


@dataclass
class ModelEvaluation:
    """Evaluation metrics for a trained model."""

    model_name: str = ""
    model_version: str = "1.0"
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    pr_auc: float = 0.0
    confusion_matrix: Optional[list] = None
    false_positive_rate: float = 0.0
    test_dataset_version: str = "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Create *path* (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)

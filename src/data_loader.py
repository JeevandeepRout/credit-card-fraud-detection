"""Dataset loading, schema validation, and summary statistics."""

from __future__ import annotations

import pandas as pd

from .utils import ALL_FEATURE_COLUMNS, EXPECTED_COLUMNS, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the credit-card fraud CSV and return a raw DataFrame.

    Raises:
        FileNotFoundError – dataset file does not exist.
        RuntimeError      – any other read failure.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n\n"
            "Please download 'creditcard.csv' from Kaggle and place it in\n"
            "the **data/** directory.\n\n"
            "URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset: {exc}") from exc
    return df


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return *(is_valid, errors)* after checking columns and dtypes."""
    errors: list[str] = []

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        errors.append(f"Unexpected columns (will be ignored): {sorted(extra)}")

    if TARGET_COLUMN in df.columns:
        unique_vals = set(df[TARGET_COLUMN].unique())
        if not unique_vals.issubset({0, 1}):
            errors.append(
                f"Target '{TARGET_COLUMN}' should contain only 0 and 1, "
                f"got: {sorted(unique_vals)}"
            )

    for col in ALL_FEATURE_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Feature '{col}' is not numeric (dtype: {df[col].dtype})")

    # "is_valid" = no *blocking* errors (missing cols, bad dtype, bad target)
    is_valid = not any(
        "Missing" in e or "not numeric" in e or "should contain" in e
        for e in errors
    )
    return is_valid, errors


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def get_summary(df: pd.DataFrame) -> dict:
    """Return high-level summary stats about the DataFrame."""
    return {
        "shape": df.shape,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(),
    }


def get_class_distribution(df: pd.DataFrame) -> dict:
    """Return class-distribution statistics for the target column."""
    if TARGET_COLUMN not in df.columns:
        return {"error": f"Target column '{TARGET_COLUMN}' not found"}

    counts = df[TARGET_COLUMN].value_counts().to_dict()
    total = len(df)
    n_fraud = int(counts.get(1, 0))
    n_legit = int(counts.get(0, 0))

    return {
        "counts": counts,
        "legitimate": n_legit,
        "fraud": n_fraud,
        "total": total,
        "fraud_percentage": round(n_fraud / total * 100, 4) if total else 0,
        "imbalance_ratio": (
            round(n_legit / n_fraud, 1) if n_fraud else float("inf")
        ),
    }

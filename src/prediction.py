"""Transaction prediction with thorough input validation."""

from __future__ import annotations

from typing import Any

from .models import load_model
from .preprocessing import load_feature_names, load_scaler, preprocess_input
from .utils import DEFAULT_THRESHOLD, TARGET_COLUMN, PredictionResult


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_input(
    features: dict[str, Any],
    feature_names: list[str],
) -> tuple[dict[str, float], list[str]]:
    """Validate a raw feature dict.

    Returns *(validated_features, errors)*.
    """
    errors: list[str] = []
    validated: dict[str, float] = {}

    if TARGET_COLUMN in features:
        errors.append(
            f"Target column '{TARGET_COLUMN}' must NOT be included in prediction input."
        )

    missing = set(feature_names) - set(features.keys())
    if missing:
        errors.append(f"Missing required features: {sorted(missing)}")
        return validated, errors

    for name in feature_names:
        value = features.get(name)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            errors.append(f"Feature '{name}' is missing or empty.")
            continue
        try:
            validated[name] = float(value)
        except (ValueError, TypeError):
            errors.append(f"Feature '{name}' must be numeric, got: '{value}'")

    return validated, errors


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_transaction(
    model_name: str,
    features: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    *,
    model=None,
    scaler=None,
    feature_names: list[str] | None = None,
) -> PredictionResult:
    """Return a :class:`PredictionResult` for one transaction.

    If *model*, *scaler*, or *feature_names* are ``None`` they are loaded
    from the ``artifacts/`` directory.
    """
    scaler = scaler or load_scaler()
    feature_names = feature_names or load_feature_names()
    model = model or load_model(model_name)

    validated, errors = validate_input(features, feature_names)
    if errors:
        raise ValueError(
            "Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    X = preprocess_input(validated, scaler, feature_names)

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        score = float(model.predict_proba(X)[:, 1][0])
        predicted_class = "Fraud" if score >= threshold else "Legitimate"
    else:
        pred = int(model.predict(X)[0])
        predicted_class = "Fraud" if pred == 1 else "Legitimate"
        score = None

    return PredictionResult(
        model_name=model_name,
        predicted_class=predicted_class,
        score=round(score, 6) if score is not None else None,
        threshold_used=threshold,
    )

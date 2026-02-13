"""Regression evaluation metrics implemented from scratch alongside sklearn verification."""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: (1/n) Σ(yᵢ − ŷᵢ)²."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: √MSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/n) Σ|yᵢ − ŷᵢ|."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination: 1 − SS_res / SS_tot."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def adjusted_r_squared(
    y_true: np.ndarray, y_pred: np.ndarray, n_features: int
) -> float:
    """Adjusted R²: accounts for the number of predictors.

    Adj-R² = 1 − [(1 − R²)(n − 1) / (n − p − 1)]
    """
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    denominator = n - n_features - 1
    if denominator <= 0:
        logger.warning("Adjusted R² undefined: n_samples (%d) <= n_features + 1 (%d)", n, n_features + 1)
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / denominator)


def aic(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Akaike Information Criterion (assuming Gaussian errors).

    AIC = n·ln(SS_res / n) + 2(p + 1)
    """
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(n * np.log(ss_res / n) + 2 * (n_features + 1))


def bic(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Bayesian Information Criterion (assuming Gaussian errors).

    BIC = n·ln(SS_res / n) + ln(n)·(p + 1)
    """
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(n * np.log(ss_res / n) + np.log(n) * (n_features + 1))


# ── Convenience ────────────────────────────────────────────────

METRIC_REGISTRY = {
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r_squared,
}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute a suite of regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.
        n_features: Number of input features (needed for Adj-R², AIC, BIC).
        metric_names: Optional list of metric names to compute.
            Defaults to all available metrics.

    Returns:
        Dictionary mapping metric names to computed values.
    """
    if metric_names is None:
        metric_names = ["mse", "rmse", "mae", "r2", "adj_r2", "aic", "bic"]

    results: dict[str, float] = {}

    for name in metric_names:
        if name in METRIC_REGISTRY:
            results[name] = METRIC_REGISTRY[name](y_true, y_pred)
        elif name == "adj_r2":
            results[name] = adjusted_r_squared(y_true, y_pred, n_features)
        elif name == "aic":
            results[name] = aic(y_true, y_pred, n_features)
        elif name == "bic":
            results[name] = bic(y_true, y_pred, n_features)
        else:
            logger.warning("Unknown metric '%s', skipping", name)

    # Log results
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in results.items())
    logger.info("Metrics — %s", metrics_str)

    return results

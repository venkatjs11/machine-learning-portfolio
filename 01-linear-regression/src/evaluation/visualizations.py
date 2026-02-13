"""Publication-quality diagnostic and evaluation plots for linear regression."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Style defaults ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

COLORS = {
    "primary": "#2563eb",
    "secondary": "#dc2626",
    "accent": "#16a34a",
    "muted": "#6b7280",
}


def plot_residuals_vs_fitted(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Residuals vs Fitted values — checks linearity and homoscedasticity.

    A well-specified model should show residuals randomly scattered around zero
    with no systematic pattern.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_pred, residuals, alpha=0.4, s=15, color=COLORS["primary"], edgecolors="none")
    ax.axhline(0, color=COLORS["secondary"], linestyle="--", linewidth=1.2)

    # LOWESS smoother for trend detection
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(residuals, y_pred, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=COLORS["secondary"], linewidth=2, label="LOWESS")
        ax.legend()
    except ImportError:
        pass

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    _maybe_save(fig, save_path, "residuals_vs_fitted.png")
    return fig


def plot_qq(
    residuals: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Q-Q plot — checks normality of residuals.

    Points should lie close to the 45° reference line if residuals are normally distributed.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    standardized = (residuals - residuals.mean()) / residuals.std()
    theoretical = np.sort(stats.norm.ppf(np.linspace(0.001, 0.999, len(standardized))))
    observed = np.sort(standardized)

    # Trim to same length if needed
    min_len = min(len(theoretical), len(observed))
    theoretical = theoretical[:min_len]
    observed = observed[:min_len]

    ax.scatter(theoretical, observed, alpha=0.4, s=15, color=COLORS["primary"], edgecolors="none")

    lim = max(abs(theoretical.min()), abs(theoretical.max()), abs(observed.min()), abs(observed.max()))
    ax.plot([-lim, lim], [-lim, lim], color=COLORS["secondary"], linestyle="--", linewidth=1.2)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Normal Q-Q Plot")
    ax.grid(True, alpha=0.3)

    _maybe_save(fig, save_path, "qq_plot.png")
    return fig


def plot_scale_location(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Scale-Location plot — checks homoscedasticity using √|standardized residuals|.

    A horizontal band indicates constant variance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    std_resid = residuals / (residuals.std() + 1e-10)
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))

    ax.scatter(y_pred, sqrt_abs_resid, alpha=0.4, s=15, color=COLORS["primary"], edgecolors="none")

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(sqrt_abs_resid, y_pred, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=COLORS["secondary"], linewidth=2, label="LOWESS")
        ax.legend()
    except ImportError:
        pass

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.set_title("Scale-Location")
    ax.grid(True, alpha=0.3)

    _maybe_save(fig, save_path, "scale_location.png")
    return fig


def plot_cooks_distance(
    cooks_d: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Cook's Distance plot — identifies influential observations.

    Points above the 4/n threshold may disproportionately affect the model.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    n = len(cooks_d)
    indices = np.arange(n)
    threshold = 4.0 / n

    colors = np.where(cooks_d > threshold, COLORS["secondary"], COLORS["primary"])
    ax.bar(indices, cooks_d, color=colors, width=1.0, alpha=0.7)
    ax.axhline(threshold, color=COLORS["secondary"], linestyle="--", linewidth=1.2, label=f"4/n = {threshold:.4f}")

    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Cook's Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _maybe_save(fig, save_path, "cooks_distance.png")
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Actual vs Predicted scatter — the closer to the diagonal, the better."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=COLORS["primary"], edgecolors="none")

    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot(
        [lim_min - margin, lim_max + margin],
        [lim_min - margin, lim_max + margin],
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=1.2,
        label="Perfect prediction",
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _maybe_save(fig, save_path, "actual_vs_predicted.png")
    return fig


def plot_coefficient_bar(
    feature_names: list[str],
    coefficients: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of model coefficients ranked by magnitude."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))

    sorted_idx = np.argsort(np.abs(coefficients))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_coefs = coefficients[sorted_idx]

    colors = [COLORS["primary"] if c >= 0 else COLORS["secondary"] for c in sorted_coefs]
    ax.barh(sorted_names, sorted_coefs, color=colors, alpha=0.8)

    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Coefficients (sorted by magnitude)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    _maybe_save(fig, save_path, "coefficients.png")
    return fig


def plot_cost_history(
    cost_history: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Gradient descent convergence curve."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(cost_history, color=COLORS["primary"], linewidth=2)
    ax.set_xlabel("Iteration (×100)")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Gradient Descent Convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    _maybe_save(fig, save_path, "cost_history.png")
    return fig


def create_diagnostic_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cooks_d: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """4-panel diagnostic dashboard (similar to R's plot.lm)."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=COLORS["primary"], edgecolors="none")
    ax.axhline(0, color=COLORS["secondary"], linestyle="--")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    # Q-Q
    ax = axes[0, 1]
    std_resid = (residuals - residuals.mean()) / (residuals.std() + 1e-10)
    stats.probplot(std_resid, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q")
    ax.grid(True, alpha=0.3)

    # Scale-Location
    ax = axes[1, 0]
    sqrt_abs = np.sqrt(np.abs(std_resid))
    ax.scatter(y_pred, sqrt_abs, alpha=0.3, s=10, color=COLORS["primary"], edgecolors="none")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("√|Std Residuals|")
    ax.set_title("Scale-Location")
    ax.grid(True, alpha=0.3)

    # Cook's Distance
    ax = axes[1, 1]
    n = len(cooks_d)
    ax.bar(range(n), cooks_d, color=COLORS["primary"], width=1.0, alpha=0.6)
    ax.axhline(4.0 / n, color=COLORS["secondary"], linestyle="--", label=f"4/n = {4.0/n:.4f}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Cook's D")
    ax.set_title("Cook's Distance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Regression Diagnostic Dashboard", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    _maybe_save(fig, save_path, "diagnostic_dashboard.png")
    return fig


# ── Helpers ────────────────────────────────────────────────────

def _maybe_save(fig: plt.Figure, save_dir: str | None, filename: str) -> None:
    """Save figure to disk if a directory is provided."""
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / filename
        fig.savefig(filepath)
        logger.info("Plot saved: %s", filepath)
    plt.close(fig)

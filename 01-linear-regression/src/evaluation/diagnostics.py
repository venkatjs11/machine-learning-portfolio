"""Statistical diagnostics for validating linear regression assumptions.

Tests covered:
- Normality of residuals (Shapiro-Wilk, Jarque-Bera)
- Homoscedasticity (Breusch-Pagan)
- Multicollinearity (Variance Inflation Factor)
- Influence analysis (Cook's Distance, Leverage)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DiagnosticReport:
    """Container for all diagnostic test results."""

    # Normality
    shapiro_stat: float
    shapiro_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float

    # Homoscedasticity
    breusch_pagan_stat: float
    breusch_pagan_pvalue: float

    # Multicollinearity
    vif_scores: dict[str, float]

    # Influence
    cooks_distance: np.ndarray
    leverage: np.ndarray
    n_influential: int

    def summary(self) -> str:
        """Human-readable diagnostic summary."""
        lines = [
            "═══ Regression Diagnostics ═══",
            "",
            "NORMALITY OF RESIDUALS",
            f"  Shapiro-Wilk     W={self.shapiro_stat:.4f}  p={self.shapiro_pvalue:.4f}  "
            f"{'✓ Normal' if self.shapiro_pvalue > 0.05 else '✗ Non-normal'}",
            f"  Jarque-Bera      JB={self.jarque_bera_stat:.4f}  p={self.jarque_bera_pvalue:.4f}  "
            f"{'✓ Normal' if self.jarque_bera_pvalue > 0.05 else '✗ Non-normal'}",
            "",
            "HOMOSCEDASTICITY",
            f"  Breusch-Pagan    LM={self.breusch_pagan_stat:.4f}  p={self.breusch_pagan_pvalue:.4f}  "
            f"{'✓ Homoscedastic' if self.breusch_pagan_pvalue > 0.05 else '✗ Heteroscedastic'}",
            "",
            "MULTICOLLINEARITY (VIF > 10 is problematic)",
        ]
        for feat, vif in sorted(self.vif_scores.items(), key=lambda x: x[1], reverse=True):
            flag = " ⚠" if vif > 10 else ""
            lines.append(f"  {feat:>25s}  VIF={vif:8.2f}{flag}")
        lines.extend([
            "",
            f"INFLUENCE: {self.n_influential} observations with Cook's D > 4/n",
        ])
        return "\n".join(lines)


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute raw residuals."""
    return y_true - y_pred


def test_normality(residuals: np.ndarray) -> tuple[float, float, float, float]:
    """Test normality of residuals using Shapiro-Wilk and Jarque-Bera.

    For large samples (>5000), Shapiro-Wilk is applied to a random subset.

    Returns:
        (shapiro_stat, shapiro_p, jb_stat, jb_p)
    """
    # Shapiro-Wilk (subsample if needed — test is O(n²))
    sample = residuals if len(residuals) <= 5000 else np.random.default_rng(42).choice(residuals, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sample)

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(residuals)

    return float(sw_stat), float(sw_p), float(jb_stat), float(jb_p)


def test_homoscedasticity(
    X: np.ndarray, residuals: np.ndarray
) -> tuple[float, float]:
    """Breusch-Pagan test for heteroscedasticity.

    Regresses squared residuals on the original features and tests
    whether the resulting R² is statistically significant.

    Returns:
        (test_statistic, p_value)
    """
    n = len(residuals)
    sq_resid = residuals ** 2
    sq_resid_norm = sq_resid / sq_resid.mean()

    # Auxiliary regression: squared residuals ~ X
    X_with_const = np.column_stack([np.ones(n), X])
    beta_aux = np.linalg.lstsq(X_with_const, sq_resid_norm, rcond=None)[0]
    fitted_aux = X_with_const @ beta_aux

    ss_reg = np.sum((fitted_aux - sq_resid_norm.mean()) ** 2)
    lm_stat = ss_reg / 2.0

    p_value = 1 - stats.chi2.cdf(lm_stat, df=X.shape[1])

    return float(lm_stat), float(p_value)


def compute_vif(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factors for each feature.

    VIFⱼ = 1 / (1 − Rⱼ²), where Rⱼ² is the R² from regressing feature j
    on all other features.

    Args:
        X: Feature matrix.
        feature_names: List of feature names.

    Returns:
        Dictionary mapping feature names to VIF values.
    """
    vif_dict: dict[str, float] = {}
    n_features = X.shape[1]

    for j in range(n_features):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        X_others_c = np.column_stack([np.ones(X.shape[0]), X_others])

        beta = np.linalg.lstsq(X_others_c, y_j, rcond=None)[0]
        y_pred = X_others_c @ beta
        ss_res = np.sum((y_j - y_pred) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)

        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else float("inf")
        vif_dict[feature_names[j]] = vif

    return vif_dict


def compute_influence(X: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cook's Distance and leverage (hat matrix diagonal).

    Args:
        X: Feature matrix.
        residuals: Model residuals.

    Returns:
        Tuple of (cooks_distance, leverage) arrays.
    """
    X_c = np.column_stack([np.ones(X.shape[0]), X])
    hat_matrix = X_c @ np.linalg.pinv(X_c.T @ X_c) @ X_c.T
    leverage = np.diag(hat_matrix)

    p = X.shape[1] + 1  # number of parameters including intercept
    mse = np.mean(residuals ** 2)

    # Cook's Distance = (eᵢ² · hᵢᵢ) / (p · MSE · (1 − hᵢᵢ)²)
    denominator = p * mse * (1 - leverage) ** 2
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    cooks_d = (residuals ** 2 * leverage) / denominator

    return cooks_d, leverage


def run_diagnostics(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> DiagnosticReport:
    """Run the full diagnostic suite.

    Args:
        X: Feature matrix used for prediction.
        y_true: Ground-truth target.
        y_pred: Model predictions.
        feature_names: Feature names for VIF reporting.

    Returns:
        A DiagnosticReport with all test results.
    """
    residuals = compute_residuals(y_true, y_pred)

    # Normality
    sw_stat, sw_p, jb_stat, jb_p = test_normality(residuals)

    # Homoscedasticity
    bp_stat, bp_p = test_homoscedasticity(X, residuals)

    # VIF
    vif_scores = compute_vif(X, feature_names)

    # Influence
    cooks_d, leverage = compute_influence(X, residuals)
    n = len(residuals)
    n_influential = int(np.sum(cooks_d > 4.0 / n))

    report = DiagnosticReport(
        shapiro_stat=sw_stat,
        shapiro_pvalue=sw_p,
        jarque_bera_stat=jb_stat,
        jarque_bera_pvalue=jb_p,
        breusch_pagan_stat=bp_stat,
        breusch_pagan_pvalue=bp_p,
        vif_scores=vif_scores,
        cooks_distance=cooks_d,
        leverage=leverage,
        n_influential=n_influential,
    )

    logger.info("\n%s", report.summary())
    return report

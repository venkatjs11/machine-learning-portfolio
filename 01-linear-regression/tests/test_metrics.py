"""Unit tests for regression evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2

from src.evaluation.metrics import (
    adjusted_r_squared,
    aic,
    bic,
    compute_all_metrics,
    mean_absolute_error,
    mean_squared_error,
    r_squared,
    root_mean_squared_error,
)


@pytest.fixture
def predictions():
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(100) * 10 + 50
    y_pred = y_true + rng.normal(0, 2, 100)
    return y_true, y_pred


class TestMetrics:
    """Verify our from-scratch metrics match scikit-learn."""

    def test_mse_matches_sklearn(self, predictions):
        y_true, y_pred = predictions
        assert abs(mean_squared_error(y_true, y_pred) - sk_mse(y_true, y_pred)) < 1e-10

    def test_rmse(self, predictions):
        y_true, y_pred = predictions
        expected = np.sqrt(sk_mse(y_true, y_pred))
        assert abs(root_mean_squared_error(y_true, y_pred) - expected) < 1e-10

    def test_mae_matches_sklearn(self, predictions):
        y_true, y_pred = predictions
        assert abs(mean_absolute_error(y_true, y_pred) - sk_mae(y_true, y_pred)) < 1e-10

    def test_r2_matches_sklearn(self, predictions):
        y_true, y_pred = predictions
        assert abs(r_squared(y_true, y_pred) - sk_r2(y_true, y_pred)) < 1e-10

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0
        assert r_squared(y, y) == 1.0

    def test_adjusted_r2_less_than_r2(self, predictions):
        y_true, y_pred = predictions
        r2 = r_squared(y_true, y_pred)
        adj_r2 = adjusted_r_squared(y_true, y_pred, n_features=5)
        assert adj_r2 <= r2

    def test_aic_bic_are_finite(self, predictions):
        y_true, y_pred = predictions
        assert np.isfinite(aic(y_true, y_pred, n_features=5))
        assert np.isfinite(bic(y_true, y_pred, n_features=5))

    def test_compute_all_returns_dict(self, predictions):
        y_true, y_pred = predictions
        results = compute_all_metrics(y_true, y_pred, n_features=3)
        assert isinstance(results, dict)
        assert "mse" in results
        assert "r2" in results

    def test_custom_metric_list(self, predictions):
        y_true, y_pred = predictions
        results = compute_all_metrics(y_true, y_pred, n_features=3, metric_names=["mse", "mae"])
        assert set(results.keys()) == {"mse", "mae"}

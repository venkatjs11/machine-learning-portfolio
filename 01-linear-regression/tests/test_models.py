"""Unit tests for from-scratch and sklearn linear regression models."""

import numpy as np
import pytest

from src.models.linear_regression_scratch import LinearRegressionScratch
from src.models.linear_regression_sklearn import LinearRegressionSklearn


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """Generate a simple linear dataset: y = 3x₁ + 2x₂ + 5 + noise."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    y = 3 * X[:, 0] + 2 * X[:, 1] + 5 + rng.normal(0, 0.1, n)
    return X, y


@pytest.fixture
def collinear_data():
    """Data with perfectly correlated features."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    X = np.column_stack([x1, x1 * 2 + 0.01 * rng.standard_normal(n)])
    y = 3 * x1 + 7 + rng.normal(0, 0.1, n)
    return X, y


# ── OLS Tests ──────────────────────────────────────────────────

class TestOLS:
    """Tests for the Normal Equation (OLS) solver."""

    def test_fits_linear_relationship(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(method="ols")
        model.fit(X, y)
        assert model.weights is not None
        assert len(model.weights) == 2

    def test_recovers_true_coefficients(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(method="ols")
        model.fit(X, y)
        np.testing.assert_allclose(model.weights, [3.0, 2.0], atol=0.15)
        np.testing.assert_allclose(model.bias, 5.0, atol=0.15)

    def test_predictions_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(method="ols").fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_r_squared_high(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(method="ols").fit(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.99

    def test_handles_collinear_features(self, collinear_data):
        """OLS with pseudoinverse should handle near-collinearity gracefully."""
        X, y = collinear_data
        model = LinearRegressionScratch(method="ols")
        model.fit(X, y)
        preds = model.predict(X)
        assert np.isfinite(preds).all()


# ── Gradient Descent Tests ─────────────────────────────────────

class TestGradientDescent:
    """Tests for the gradient descent solver."""

    def test_converges(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(
            method="gradient_descent",
            learning_rate=0.05,
            max_iterations=5000,
        )
        model.fit(X, y)
        assert len(model.cost_history) > 0
        assert model.cost_history[-1] < model.cost_history[0]

    def test_recovers_coefficients(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(
            method="gradient_descent",
            learning_rate=0.05,
            max_iterations=10000,
        ).fit(X, y)
        np.testing.assert_allclose(model.weights, [3.0, 2.0], atol=0.3)
        np.testing.assert_allclose(model.bias, 5.0, atol=0.3)

    def test_mini_batch(self, simple_data):
        X, y = simple_data
        model = LinearRegressionScratch(
            method="gradient_descent",
            learning_rate=0.01,
            max_iterations=5000,
            batch_size=64,
        ).fit(X, y)
        preds = model.predict(X)
        assert np.isfinite(preds).all()


# ── Sklearn Wrapper Tests ──────────────────────────────────────

class TestSklearnWrapper:
    """Tests for the scikit-learn model wrapper."""

    @pytest.fixture
    def ols_config(self):
        return {
            "model": {"method": "ols", "regularization": {}},
        }

    @pytest.fixture
    def ridge_config(self):
        return {
            "model": {
                "method": "ridge",
                "regularization": {
                    "alpha_search": [0.01, 0.1, 1.0, 10.0],
                    "cv_folds": 3,
                },
            },
        }

    def test_sklearn_ols(self, simple_data, ols_config):
        X, y = simple_data
        model = LinearRegressionSklearn(ols_config).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_sklearn_ridge(self, simple_data, ridge_config):
        X, y = simple_data
        model = LinearRegressionSklearn(ridge_config).fit(X, y)
        assert model.best_params is not None
        assert "alpha" in model.best_params

    def test_feature_importance(self, simple_data, ols_config):
        X, y = simple_data
        model = LinearRegressionSklearn(ols_config).fit(X, y)
        importance = model.get_feature_importance(["x1", "x2"])
        assert len(importance) == 2
        # x1 (coef=3) should be more important than x2 (coef=2)
        assert importance[0][0] == "x1"


# ── Edge Cases ─────────────────────────────────────────────────

class TestEdgeCases:
    """Validation and error handling tests."""

    def test_predict_before_fit_raises(self):
        model = LinearRegressionScratch(method="ols")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([[1, 2]]))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="must be"):
            LinearRegressionScratch(method="random_forest")

    def test_1d_input_raises(self):
        model = LinearRegressionScratch(method="ols")
        with pytest.raises(ValueError, match="2-dimensional"):
            model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_mismatched_samples_raises(self):
        model = LinearRegressionScratch(method="ols")
        with pytest.raises(ValueError, match="same number"):
            model.fit(np.array([[1], [2]]), np.array([1, 2, 3]))

    def test_single_feature(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 1))
        y = 2 * X.ravel() + 1
        model = LinearRegressionScratch(method="ols").fit(X, y)
        np.testing.assert_allclose(model.weights, [2.0], atol=0.01)

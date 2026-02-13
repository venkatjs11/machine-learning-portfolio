"""Scikit-learn Linear Regression wrapper with regularization variants and hyperparameter search."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "ols": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}


class LinearRegressionSklearn:
    """Unified interface for scikit-learn linear models with optional hyperparameter tuning.

    Supports plain OLS, Ridge (L2), Lasso (L1), and ElasticNet (L1 + L2)
    with cross-validated alpha search.

    Attributes:
        model_type: Name of the regression variant.
        model: The underlying fitted scikit-learn estimator.
        best_params: Best hyperparameters found by grid search (if applicable).
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Full configuration dictionary containing 'model' section.
        """
        model_cfg = config["model"]
        self.model_type: str = model_cfg["method"]
        self._reg_cfg: dict = model_cfg.get("regularization", {})
        self.model: Any = None
        self.best_params: dict | None = None

        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{self.model_type}'. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionSklearn":
        """Fit the model, optionally running cross-validated hyperparameter search.

        For OLS, fitting is direct. For regularized variants, a grid search
        over alpha values is performed using k-fold cross-validation.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Training target vector of shape (n_samples,).

        Returns:
            self (fitted model).
        """
        if self.model_type == "ols":
            self.model = LinearRegression()
            self.model.fit(X, y)
        else:
            self.model = self._fit_with_tuning(X, y)

        logger.info("Fitted %s model (sklearn)", self.model_type)
        if self.best_params:
            logger.info("Best hyperparameters: %s", self.best_params)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Return feature importances ranked by absolute coefficient magnitude.

        Args:
            feature_names: List of feature names corresponding to columns of X.

        Returns:
            List of (feature_name, coefficient) tuples sorted by |coefficient| descending.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        coefs = self.model.coef_
        importance = sorted(
            zip(feature_names, coefs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return importance

    def _fit_with_tuning(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Run grid search over regularization strength.

        Args:
            X: Training features.
            y: Training target.

        Returns:
            Best fitted estimator from cross-validation.
        """
        model_cls = MODEL_REGISTRY[self.model_type]
        param_grid: dict[str, list] = {"alpha": self._reg_cfg.get("alpha_search", [1.0])}

        if self.model_type == "elasticnet":
            l1_ratio = self._reg_cfg.get("l1_ratio", 0.5)
            param_grid["l1_ratio"] = [l1_ratio] if isinstance(l1_ratio, float) else l1_ratio

        cv_folds = self._reg_cfg.get("cv_folds", 5)

        grid = GridSearchCV(
            estimator=model_cls(),
            param_grid=param_grid,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X, y)

        self.best_params = grid.best_params_
        return grid.best_estimator_

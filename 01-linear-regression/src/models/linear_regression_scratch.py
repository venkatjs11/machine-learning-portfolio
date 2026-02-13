"""Linear Regression implemented from scratch using NumPy.

Provides two estimation strategies:
1. Ordinary Least Squares (OLS) via the Normal Equation.
2. (Mini-Batch) Gradient Descent with convergence tracking.

Both methods are fully vectorized with no Python loops over samples.
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LinearRegressionScratch:
    """Pure-NumPy Linear Regression supporting OLS and Gradient Descent.

    Attributes:
        method: Estimation method — "ols" or "gradient_descent".
        weights: Learned coefficient vector (set after fitting).
        bias: Learned intercept term (set after fitting).
        cost_history: Per-iteration MSE loss (gradient descent only).
    """

    def __init__(
        self,
        method: str = "ols",
        learning_rate: float = 0.01,
        max_iterations: int = 10_000,
        tolerance: float = 1e-7,
        batch_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            method: "ols" for Normal Equation, "gradient_descent" for iterative optimization.
            learning_rate: Step size for gradient descent.
            max_iterations: Upper bound on gradient descent iterations.
            tolerance: Convergence threshold (absolute change in loss).
            batch_size: Samples per mini-batch. None = full-batch gradient descent.
            random_state: Seed for reproducibility in mini-batch sampling.
        """
        if method not in ("ols", "gradient_descent"):
            raise ValueError(f"method must be 'ols' or 'gradient_descent', got '{method}'")

        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.cost_history: list[float] = []

    # ── Public API ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            self (fitted model).
        """
        X, y = self._validate_inputs(X, y)

        if self.method == "ols":
            self._fit_ols(X, y)
        else:
            self._fit_gradient_descent(X, y)

        logger.info(
            "Fitted %s | method=%s | n_features=%d",
            self.__class__.__name__,
            self.method,
            X.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for new data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias

    def get_coefficients(self) -> dict[str, float | np.ndarray]:
        """Return the learned parameters.

        Returns:
            Dictionary with 'bias' (float) and 'weights' (ndarray).
        """
        return {"bias": self.bias, "weights": self.weights}

    # ── Private Methods ────────────────────────────────────────────

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        """Closed-form solution via the Normal Equation: β = (XᵀX)⁻¹Xᵀy.

        Uses the pseudoinverse for numerical stability.
        """
        # Add bias column
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        # β = pinv(X) @ y  (numerically stable)
        theta = np.linalg.pinv(X_b) @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Iterative optimization using (mini-batch) gradient descent.

        Gradients:
            ∂L/∂w = (2/n) Xᵀ(Xw + b − y)
            ∂L/∂b = (2/n) Σ(Xw + b − y)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # Initialize weights with small random values
        self.weights = rng.standard_normal(n_features) * 0.01
        self.bias = 0.0
        self.cost_history = []

        batch_size = self.batch_size or n_samples

        for iteration in range(1, self.max_iterations + 1):
            # Sample mini-batch
            if batch_size < n_samples:
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch, y_batch = X[indices], y[indices]
            else:
                X_batch, y_batch = X, y

            # Forward pass
            y_pred = X_batch @ self.weights + self.bias
            residuals = y_pred - y_batch

            # Compute gradients
            dw = (2.0 / batch_size) * (X_batch.T @ residuals)
            db = (2.0 / batch_size) * residuals.sum()

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track cost (full-batch MSE for consistent monitoring)
            if iteration % 100 == 0 or iteration == 1:
                cost = np.mean((X @ self.weights + self.bias - y) ** 2)
                self.cost_history.append(cost)

                # Convergence check
                if len(self.cost_history) >= 2:
                    delta = abs(self.cost_history[-2] - self.cost_history[-1])
                    if delta < self.tolerance:
                        logger.info("Converged at iteration %d (Δcost=%.2e)", iteration, delta)
                        return

        logger.warning(
            "Gradient descent did not converge within %d iterations (final cost=%.6f)",
            self.max_iterations,
            self.cost_history[-1] if self.cost_history else float("nan"),
        )

    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and cast inputs to float64."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples: {X.shape[0]} != {y.shape[0]}"
            )
        return X, y

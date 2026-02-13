"""Feature engineering and preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.loader import Dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class Preprocessor:
    """Stateful preprocessing pipeline that fits on training data and transforms both splits.

    Handles missing value imputation, optional outlier removal, and feature scaling.
    Follows the fit/transform pattern to prevent data leakage.

    Attributes:
        config: Preprocessing configuration dictionary.
        scaler: Fitted scaler instance (or None if scaling is disabled).
        fill_values: Per-feature fill values computed from the training set.
    """

    def __init__(self, config: dict) -> None:
        self.config = config["preprocessing"]
        self.scaler = None
        self.fill_values: np.ndarray | None = None

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fit on training data and transform both train and test sets.

        Args:
            dataset: Input Dataset with raw features.

        Returns:
            New Dataset with preprocessed features.
        """
        X_train = dataset.X_train.copy()
        X_test = dataset.X_test.copy()
        y_train = dataset.y_train.copy()
        y_test = dataset.y_test.copy()

        # ── Handle missing values ──────────────────────────────────────
        X_train, y_train = self._handle_missing(X_train, y_train, fit=True)
        X_test, y_test = self._handle_missing(X_test, y_test, fit=False)

        # ── Remove outliers (training only) ────────────────────────────
        if self.config.get("remove_outliers", False):
            threshold = self.config.get("outlier_threshold", 3.0)
            mask = self._outlier_mask(X_train, threshold)
            n_removed = (~mask).sum()
            X_train = X_train[mask]
            y_train = y_train[mask]
            logger.info("Removed %d outlier samples (z > %.1f)", n_removed, threshold)

        # ── Scale features ─────────────────────────────────────────────
        if self.config.get("scale_features", True):
            scaler_name = self.config.get("scaler", "standard")
            scaler_cls = SCALERS.get(scaler_name)
            if scaler_cls is None:
                raise ValueError(f"Unknown scaler: {scaler_name}. Choose from {list(SCALERS)}")

            self.scaler = scaler_cls()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            logger.info("Applied %s scaling", scaler_name)

        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
        )

    def _handle_missing(
        self, X: np.ndarray, y: np.ndarray, fit: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Impute or drop missing values.

        Args:
            X: Feature matrix.
            y: Target vector.
            fit: If True, compute fill values from X (training mode).

        Returns:
            Tuple of (imputed X, corresponding y).
        """
        if not np.isnan(X).any():
            return X, y

        strategy = self.config.get("handle_missing", "median")

        if strategy == "drop":
            mask = ~np.isnan(X).any(axis=1)
            logger.info("Dropped %d rows with missing values", (~mask).sum())
            return X[mask], y[mask]

        # Imputation
        if fit:
            if strategy == "mean":
                self.fill_values = np.nanmean(X, axis=0)
            elif strategy == "median":
                self.fill_values = np.nanmedian(X, axis=0)
            else:
                raise ValueError(f"Unknown missing value strategy: {strategy}")

        assert self.fill_values is not None, "Preprocessor must be fitted before transform"
        nan_mask = np.isnan(X)
        for col_idx in range(X.shape[1]):
            X[nan_mask[:, col_idx], col_idx] = self.fill_values[col_idx]

        logger.info("Imputed missing values using %s strategy", strategy)
        return X, y

    @staticmethod
    def _outlier_mask(X: np.ndarray, threshold: float) -> np.ndarray:
        """Create a boolean mask identifying non-outlier rows by Z-score.

        Args:
            X: Feature matrix.
            threshold: Z-score threshold.

        Returns:
            Boolean mask where True = keep.
        """
        z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))
        return (z_scores < threshold).all(axis=1)

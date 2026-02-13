"""Data loading, validation, and train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Dataset:
    """Container for a train/test split with metadata."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str

    @property
    def n_samples_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_samples_test(self) -> int:
        return self.X_test.shape[0]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def summary(self) -> str:
        return (
            f"Dataset: {self.n_features} features, "
            f"{self.n_samples_train} train / {self.n_samples_test} test samples, "
            f"target='{self.target_name}'"
        )


def load_sklearn_dataset() -> tuple[pd.DataFrame, str]:
    """Load the California Housing dataset as a pandas DataFrame.

    Returns:
        Tuple of (DataFrame with features + target, target column name).
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # type: ignore[attr-defined]
    target_col = "MedHouseVal"
    logger.info("Loaded California Housing dataset: %d rows, %d columns", *df.shape)
    return df, target_col


def load_csv_dataset(path: str, target_column: str) -> tuple[pd.DataFrame, str]:
    """Load a dataset from a CSV file.

    Args:
        path: Path to the CSV file.
        target_column: Name of the target variable column.

    Returns:
        Tuple of (DataFrame, target column name).

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError: If the target column is not in the DataFrame.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("Loaded CSV dataset from %s: %d rows, %d columns", path, *df.shape)
    return df, target_column


def validate_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Run basic data quality checks and log warnings.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.

    Returns:
        The validated DataFrame (unchanged).
    """
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        cols_with_missing = missing[missing > 0]
        logger.warning("Missing values detected:\n%s", cols_with_missing.to_string())

    # Check for constant features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_column, errors="ignore")
    constant_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if constant_cols:
        logger.warning("Constant features (consider removing): %s", constant_cols)

    # Check target distribution
    target = df[target_column]
    logger.info(
        "Target '%s' stats — mean: %.3f, std: %.3f, min: %.3f, max: %.3f",
        target_column,
        target.mean(),
        target.std(),
        target.min(),
        target.max(),
    )

    return df


def create_dataset(config: dict) -> Dataset:
    """End-to-end data loading and splitting driven by configuration.

    Args:
        config: Configuration dictionary with 'data' section.

    Returns:
        A Dataset object ready for modeling.
    """
    data_cfg = config["data"]

    # Load data
    if data_cfg["source"] == "sklearn":
        df, target_col = load_sklearn_dataset()
    elif data_cfg["source"] == "csv":
        df, target_col = load_csv_dataset(data_cfg["csv_path"], data_cfg["target_column"])
    else:
        raise ValueError(f"Unknown data source: {data_cfg['source']}")

    # Validate
    df = validate_dataframe(df, target_col)

    # Split features and target
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )

    dataset = Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        target_name=target_col,
    )
    logger.info(dataset.summary())
    return dataset

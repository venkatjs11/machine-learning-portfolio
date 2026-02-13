"""Unit tests for the preprocessing pipeline."""

import numpy as np
import pytest

from src.data.loader import Dataset
from src.data.preprocessor import Preprocessor


@pytest.fixture
def sample_dataset():
    rng = np.random.default_rng(42)
    n_train, n_test = 100, 30
    X_train = rng.standard_normal((n_train, 3)) * 10 + 50
    X_test = rng.standard_normal((n_test, 3)) * 10 + 50
    y_train = rng.standard_normal(n_train)
    y_test = rng.standard_normal(n_test)
    return Dataset(X_train, X_test, y_train, y_test, ["a", "b", "c"], "target")


@pytest.fixture
def config_standard():
    return {"preprocessing": {"scale_features": True, "scaler": "standard", "handle_missing": "median"}}


@pytest.fixture
def config_no_scale():
    return {"preprocessing": {"scale_features": False, "handle_missing": "median"}}


class TestPreprocessor:
    def test_standard_scaling_zero_mean(self, sample_dataset, config_standard):
        p = Preprocessor(config_standard)
        result = p.fit_transform(sample_dataset)
        np.testing.assert_allclose(result.X_train.mean(axis=0), 0, atol=1e-10)

    def test_standard_scaling_unit_var(self, sample_dataset, config_standard):
        p = Preprocessor(config_standard)
        result = p.fit_transform(sample_dataset)
        np.testing.assert_allclose(result.X_train.std(axis=0), 1, atol=1e-10)

    def test_no_scaling_preserves_data(self, sample_dataset, config_no_scale):
        p = Preprocessor(config_no_scale)
        result = p.fit_transform(sample_dataset)
        np.testing.assert_array_equal(result.X_train, sample_dataset.X_train)

    def test_missing_value_imputation(self, config_standard):
        X_train = np.array([[1, 2], [3, np.nan], [5, 6]], dtype=float)
        X_test = np.array([[np.nan, 4]], dtype=float)
        ds = Dataset(X_train, X_test, np.array([1, 2, 3.0]), np.array([1.0]), ["a", "b"], "t")

        p = Preprocessor(config_standard)
        result = p.fit_transform(ds)
        assert not np.isnan(result.X_train).any()
        assert not np.isnan(result.X_test).any()

    def test_outlier_removal(self, sample_dataset):
        cfg = {"preprocessing": {"scale_features": False, "handle_missing": "median", "remove_outliers": True, "outlier_threshold": 2.0}}
        p = Preprocessor(cfg)
        result = p.fit_transform(sample_dataset)
        assert result.X_train.shape[0] <= sample_dataset.X_train.shape[0]

    def test_shapes_preserved(self, sample_dataset, config_standard):
        p = Preprocessor(config_standard)
        result = p.fit_transform(sample_dataset)
        assert result.X_train.shape[1] == sample_dataset.X_train.shape[1]
        assert result.X_test.shape[1] == sample_dataset.X_test.shape[1]

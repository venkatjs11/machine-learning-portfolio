"""Integration tests for the FastAPI prediction endpoint."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import _state, app
from src.models.linear_regression_scratch import LinearRegressionScratch


@pytest.fixture(autouse=True)
def mock_model():
    """Inject a trained model into the API state for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = X @ np.array([1, 2, 3]) + 5
    model = LinearRegressionScratch(method="ols").fit(X, y)

    _state["model"] = model
    _state["metadata"] = {"model_name": "test_model"}
    _state["feature_names"] = ["f1", "f2", "f3"]
    yield
    _state["model"] = None


@pytest.fixture
def client():
    return TestClient(app)


class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True

    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_features"] == 3
        assert data["feature_names"] == ["f1", "f2", "f3"]

    def test_predict_single(self, client):
        resp = client.post("/predict", json={"features": [[1.0, 2.0, 3.0]]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1
        assert data["n_samples"] == 1

    def test_predict_batch(self, client):
        features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200
        assert resp.json()["n_samples"] == 3

    def test_predict_wrong_features(self, client):
        resp = client.post("/predict", json={"features": [[1.0, 2.0]]})  # expects 3
        assert resp.status_code == 422

    def test_predict_empty_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_accuracy(self, client):
        """Predictions from API should match direct model calls."""
        resp = client.post("/predict", json={"features": [[1.0, 2.0, 3.0]]})
        prediction = resp.json()["predictions"][0]
        expected = 1 * 1 + 2 * 2 + 3 * 3 + 5  # = 19
        assert abs(prediction - expected) < 0.1

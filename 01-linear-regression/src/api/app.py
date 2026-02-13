"""FastAPI REST API for serving the trained linear regression model.

Endpoints:
    GET  /health         — Health check
    GET  /model/info     — Model metadata and feature list
    POST /predict        — Single or batch prediction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.io import load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Linear Regression API",
    description="Production model serving endpoint for linear regression predictions.",
    version="1.0.0",
)


# ── Request / Response schemas ─────────────────────────────────

class PredictionRequest(BaseModel):
    """Input schema for prediction requests."""

    features: list[list[float]] = Field(
        ...,
        description="2D array of feature values. Each inner list is one observation.",
        min_length=1,
        json_schema_extra={"examples": [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]},
    )


class PredictionResponse(BaseModel):
    """Output schema for predictions."""

    predictions: list[float]
    n_samples: int


class ModelInfo(BaseModel):
    """Model metadata schema."""

    model_name: str
    n_features: int
    feature_names: list[str]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check schema."""

    status: str
    model_loaded: bool


# ── Model state ────────────────────────────────────────────────

_state: dict[str, Any] = {
    "model": None,
    "metadata": None,
    "feature_names": None,
}


def load_latest_model(model_dir: str = "outputs/models") -> None:
    """Load the most recently saved model and its metadata.

    Args:
        model_dir: Directory containing model artifacts.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.warning("Model directory does not exist: %s", model_dir)
        return

    # Find latest model file
    model_files = sorted(model_path.glob("*.joblib"), reverse=True)
    if not model_files:
        logger.warning("No model files found in %s", model_dir)
        return

    latest = model_files[0]
    _state["model"] = load_model(str(latest))

    # Load companion metadata
    meta_file = latest.with_name(latest.stem + "_meta.json")
    if meta_file.exists():
        _state["metadata"] = json.loads(meta_file.read_text())
    else:
        _state["metadata"] = {"model_file": latest.name}

    logger.info("API loaded model: %s", latest.name)


# ── Endpoints ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    """Attempt to load model on API startup."""
    load_latest_model()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Return model metadata and feature list."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    model = _state["model"]
    metadata = _state["metadata"] or {}

    # Infer feature count from model weights
    n_features = (
        len(model.weights)
        if hasattr(model, "weights") and model.weights is not None
        else getattr(model, "n_features_in_", 0)
    )

    return ModelInfo(
        model_name=metadata.get("model_name", "linear_regression"),
        n_features=n_features,
        feature_names=_state.get("feature_names") or [f"x{i}" for i in range(n_features)],
        metadata=metadata,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate predictions for one or more observations.

    Args:
        request: JSON body containing a 2D feature array.

    Returns:
        Predicted values and count.
    """
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first.")

    try:
        X = np.array(request.features, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid feature data: {e}")

    # Validate feature dimension
    model = _state["model"]
    expected_features = (
        len(model.weights)
        if hasattr(model, "weights") and model.weights is not None
        else getattr(model, "n_features_in_", X.shape[1])
    )
    if X.shape[1] != expected_features:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_features} features, got {X.shape[1]}",
        )

    predictions = model.predict(X)
    return PredictionResponse(
        predictions=predictions.tolist(),
        n_samples=len(predictions),
    )

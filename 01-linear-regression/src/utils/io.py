"""Model serialization and metadata management."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: Any,
    metrics: dict[str, float],
    config: dict,
    output_dir: str = "outputs/models",
    model_name: str = "linear_regression",
) -> Path:
    """Save a trained model with metadata for reproducibility.

    Persists the model artifact via joblib and writes a companion JSON file
    containing evaluation metrics, configuration hash, and timestamp.

    Args:
        model: Trained model object.
        metrics: Dictionary of evaluation metrics.
        config: Training configuration dictionary.
        output_dir: Directory to save model artifacts.
        model_name: Base name for the saved files.

    Returns:
        Path to the saved model file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"{model_name}_{timestamp}.joblib"
    meta_filename = f"{model_name}_{timestamp}_meta.json"

    model_path = out / model_filename
    meta_path = out / meta_filename

    # Save model
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # Save metadata
    config_hash = hashlib.sha256(yaml.dump(config).encode()).hexdigest()[:12]
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "metrics": metrics,
        "model_file": model_filename,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata saved to %s", meta_path)

    return model_path


def load_model(model_path: str) -> Any:
    """Load a persisted model from disk.

    Args:
        model_path: Path to the joblib model file.

    Returns:
        Deserialized model object.
    """
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model

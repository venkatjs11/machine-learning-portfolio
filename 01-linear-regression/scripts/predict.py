"""CLI entrypoint for batch prediction using a trained model.

Usage:
    python scripts/predict.py --model outputs/models/latest.joblib --input data.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_model
from src.utils.logger import get_logger

logger = get_logger("predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prediction with a trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.joblib)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load input data
    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # Predict
    X = df.select_dtypes(include=[np.number]).values
    predictions = model.predict(X)

    # Save
    output_df = df.copy()
    output_df["prediction"] = predictions
    output_df.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    main()

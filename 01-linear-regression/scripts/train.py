"""CLI entrypoint for training a linear regression model.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import create_dataset
from src.data.preprocessor import Preprocessor
from src.evaluation.diagnostics import run_diagnostics
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualizations import (
    create_diagnostic_dashboard,
    plot_actual_vs_predicted,
    plot_coefficient_bar,
    plot_cost_history,
)
from src.models.linear_regression_scratch import LinearRegressionScratch
from src.models.linear_regression_sklearn import LinearRegressionSklearn
from src.utils.io import save_model
from src.utils.logger import get_logger


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear regression model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--diagnostics", action="store_true", help="Run full diagnostic suite")
    args = parser.parse_args()

    config = load_config(args.config)
    log_file = config.get("output", {}).get("log_file")
    logger = get_logger("train", log_file=log_file)

    logger.info("=" * 60)
    logger.info("LINEAR REGRESSION TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load & preprocess ───────────────────────────────────
    logger.info("Step 1: Loading data...")
    dataset = create_dataset(config)

    logger.info("Step 2: Preprocessing...")
    preprocessor = Preprocessor(config)
    dataset = preprocessor.fit_transform(dataset)

    # ── 2. Train model ─────────────────────────────────────────
    method = config["model"]["method"]
    logger.info("Step 3: Training model (method=%s)...", method)

    if method in ("ols", "gradient_descent"):
        gd_cfg = config["model"].get("gradient_descent", {})
        model = LinearRegressionScratch(
            method=method,
            learning_rate=gd_cfg.get("learning_rate", 0.01),
            max_iterations=gd_cfg.get("max_iterations", 10000),
            tolerance=gd_cfg.get("tolerance", 1e-7),
            batch_size=gd_cfg.get("batch_size"),
            random_state=config["data"].get("random_state", 42),
        )
        model.fit(dataset.X_train, dataset.y_train)
    else:
        model = LinearRegressionSklearn(config)
        model.fit(dataset.X_train, dataset.y_train)

    # ── 3. Evaluate ────────────────────────────────────────────
    logger.info("Step 4: Evaluating...")

    y_pred_train = model.predict(dataset.X_train)
    y_pred_test = model.predict(dataset.X_test)

    logger.info("--- Training Set ---")
    train_metrics = compute_all_metrics(
        dataset.y_train, y_pred_train, dataset.n_features,
        metric_names=config["evaluation"]["metrics"],
    )

    logger.info("--- Test Set ---")
    test_metrics = compute_all_metrics(
        dataset.y_test, y_pred_test, dataset.n_features,
        metric_names=config["evaluation"]["metrics"],
    )

    # ── 4. Diagnostics ─────────────────────────────────────────
    plots_dir = config["evaluation"].get("plots_dir", "outputs/plots")

    if args.diagnostics or config["evaluation"].get("diagnostics", False):
        logger.info("Step 5: Running diagnostics...")
        report = run_diagnostics(
            dataset.X_test, dataset.y_test, y_pred_test, dataset.feature_names
        )

        # Diagnostic dashboard
        create_diagnostic_dashboard(
            dataset.y_test, y_pred_test, report.cooks_distance, save_path=plots_dir
        )

    # ── 5. Plots ───────────────────────────────────────────────
    if config["evaluation"].get("save_plots", True):
        logger.info("Generating plots...")
        plot_actual_vs_predicted(dataset.y_test, y_pred_test, save_path=plots_dir)

        # Coefficients
        weights = (
            model.weights if hasattr(model, "weights") and model.weights is not None
            else model.model.coef_
        )
        plot_coefficient_bar(dataset.feature_names, weights, save_path=plots_dir)

        # Cost history (gradient descent only)
        if hasattr(model, "cost_history") and model.cost_history:
            plot_cost_history(model.cost_history, save_path=plots_dir)

    # ── 6. Save model ──────────────────────────────────────────
    model_dir = config["output"]["model_dir"]
    save_model(model, test_metrics, config, output_dir=model_dir)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

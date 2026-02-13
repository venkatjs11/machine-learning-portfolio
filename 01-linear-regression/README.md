# 📈 Linear Regression — From Scratch to Production

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade implementation of Linear Regression covering **theory, implementation from scratch, scikit-learn benchmarking, rigorous diagnostics, REST API serving, and full MLOps practices**. Built to demonstrate end-to-end ML engineering — not just modeling.

---

## 🏗️ Project Structure

```
linear-regression/
├── configs/                  # Hydra-style YAML configurations
│   └── config.yaml
├── docs/                     # Additional documentation & math derivations
│   └── theory.md
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── scripts/
│   ├── train.py              # CLI training entrypoint
│   └── predict.py            # CLI batch prediction
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Data loading & validation
│   │   └── preprocessor.py   # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear_regression_scratch.py   # NumPy-only OLS + Gradient Descent
│   │   └── linear_regression_sklearn.py   # Scikit-learn wrapper with regularization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # Regression metrics (MSE, RMSE, MAE, R², Adj-R²)
│   │   ├── diagnostics.py    # Residual analysis, VIF, heteroscedasticity tests
│   │   └── visualizations.py # Publication-quality diagnostic plots
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py            # FastAPI model-serving endpoint
│   └── utils/
│       ├── __init__.py
│       ├── logger.py         # Structured logging
│       └── io.py             # Model serialization (joblib + JSON metadata)
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_preprocessor.py
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI pipeline
├── .gitignore
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Clone & set up
git clone https://github.com/<your-username>/linear-regression.git
cd linear-regression
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train a model
python scripts/train.py --config configs/config.yaml

# Run diagnostics
python scripts/train.py --config configs/config.yaml --diagnostics

# Serve the model
uvicorn src.api.app:app --reload

# Run tests
pytest tests/ -v --cov=src
```

## 🔬 What's Inside

### 1. Implementation from Scratch (`src/models/linear_regression_scratch.py`)
- **Ordinary Least Squares** via the Normal Equation: `β = (XᵀX)⁻¹Xᵀy`
- **Gradient Descent** with configurable learning rate, convergence tolerance, and max iterations
- **Mini-Batch Gradient Descent** for scalability demonstrations
- Cost history tracking for convergence visualization
- Full NumPy vectorization — no loops over samples

### 2. Scikit-learn Benchmarking (`src/models/linear_regression_sklearn.py`)
- Unified interface wrapping `LinearRegression`, `Ridge`, `Lasso`, and `ElasticNet`
- Hyperparameter tuning via cross-validated grid search
- Feature importance extraction and ranking

### 3. Rigorous Diagnostics (`src/evaluation/`)
- **Residual analysis**: normality (Shapiro-Wilk, Jarque-Bera), homoscedasticity (Breusch-Pagan)
- **Multicollinearity detection**: Variance Inflation Factor (VIF)
- **Influence analysis**: Cook's Distance, Leverage (hat matrix)
- **Diagnostic plots**: Residuals vs Fitted, Q-Q, Scale-Location, Cook's Distance

### 4. Production Serving (`src/api/app.py`)
- FastAPI REST endpoint with Pydantic request/response validation
- Health check and model metadata endpoints
- Input feature validation and error handling

### 5. MLOps & Engineering Practices
- Typed configuration via YAML + dataclasses
- Structured logging with rotation
- Model versioning with JSON metadata (metrics, timestamp, config hash)
- Reproducible with random seed control
- 90%+ test coverage target

## 📊 Key Metrics Tracked

| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |
| Adjusted R² | R² corrected for number of predictors |
| AIC / BIC | Information criteria for model comparison |

## 🧪 Testing

```bash
# Unit tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Type checking
mypy src/ --ignore-missing-imports

# Linting
ruff check src/ tests/
```

## 🐳 Docker

```bash
docker build -t linear-regression .
docker run -p 8000:8000 linear-regression
```

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) — theoretical foundations
- [scikit-learn documentation](https://scikit-learn.org/) — reference implementations
- [FastAPI](https://fastapi.tiangolo.com/) — API framework

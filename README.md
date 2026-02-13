# machine-learning-portfolio
"Production-grade ML algorithms — from scratch to deployment"
# 🧠 Machine Learning From Scratch — Production Grade

End-to-end implementations of core ML algorithms, each built from scratch
with NumPy, benchmarked against scikit-learn, fully tested, and served via
REST APIs.

Every module follows the same production pattern:
**Theory → From-Scratch Implementation → sklearn Benchmarking → Diagnostics → API Serving → Tests → CI/CD**

---

## 📚 Algorithms

| # | Algorithm | Key Concepts | Status |
|---|-----------|-------------|--------|
| 01 | [Linear Regression](./01-linear-regression/) | OLS, Gradient Descent, Ridge, Lasso, ElasticNet | ✅ Complete |
| 02 | [Logistic Regression](./02-logistic-regression/) | Sigmoid, Cross-Entropy, Regularization | 🔜 Coming |
| 03 | [Decision Trees](./03-decision-trees/) | Gini, Entropy, Pruning, Feature Importance | 🔜 Coming |
| 04 | [K-Nearest Neighbors](./04-knn/) | Distance Metrics, KD-Trees, Curse of Dimensionality | 🔜 Coming |
| 05 | [Support Vector Machines](./05-svm/) | Kernel Trick, Margin Maximization, SMO | 🔜 Coming |
| 06 | [Neural Network](./06-neural-network/) | Backpropagation, Activations, Batch Norm | 🔜 Coming |

---

## 🏗️ Consistent Structure

Each algorithm folder follows this layout:
```
XX-algorithm-name/
├── src/               # Core implementation (from scratch + sklearn)
│   ├── models/        # Algorithm implementations
│   ├── data/          # Loading & preprocessing
│   ├── evaluation/    # Metrics, diagnostics, plots
│   └── api/           # FastAPI serving endpoint
├── tests/             # Pytest suite (80%+ coverage)
├── notebooks/         # EDA & walkthrough notebook
├── docs/              # Mathematical derivations
├── configs/           # YAML configuration
├── scripts/           # CLI entrypoints (train, predict)
├── Dockerfile         # Container deployment
└── Makefile           # Common dev commands
```

## 🚀 Quick Start (any algorithm)
```bash
cd XX-algorithm-name/
pip install -r requirements.txt
make train          # Train the model
make test           # Run test suite
make serve          # Start API server
```

## 👤 Author

**Your Name** — [LinkedIn](https://linkedin.com/in/yourprofile) | [Email](mailto:you@example.com)

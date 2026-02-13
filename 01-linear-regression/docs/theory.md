# Linear Regression — Mathematical Foundations

## 1. Problem Formulation

Given a dataset of *n* observations with *p* features:

- **Feature matrix**: X ∈ ℝⁿˣᵖ
- **Target vector**: y ∈ ℝⁿ
- **Parameter vector**: β ∈ ℝᵖ, intercept β₀ ∈ ℝ

The linear model assumes: **ŷ = Xβ + β₀**

## 2. Ordinary Least Squares (OLS)

### Objective

Minimize the sum of squared residuals:

**L(β) = ‖y − Xβ‖² = (y − Xβ)ᵀ(y − Xβ)**

### Normal Equation (Closed-Form Solution)

Setting ∂L/∂β = 0 yields:

**β̂ = (XᵀX)⁻¹Xᵀy**

This implementation uses the Moore-Penrose pseudoinverse (`np.linalg.pinv`) for numerical stability when XᵀX is ill-conditioned.

**Time complexity**: O(np² + p³) — dominated by matrix inversion.

### Assumptions

1. **Linearity**: E[y | X] is a linear function of X
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Var(εᵢ) = σ² (constant variance)
4. **Normality**: ε ~ N(0, σ²I) (for inference)
5. **No perfect multicollinearity**: rank(X) = p

## 3. Gradient Descent

### Update Rule

For learning rate η:

**β ← β − η · ∂L/∂β**

Where the gradient is:

**∂L/∂β = (2/n) · Xᵀ(Xβ − y)**

### Mini-Batch Variant

At each iteration, sample a subset B ⊂ {1, …, n}:

**∂L/∂β ≈ (2/|B|) · X_Bᵀ(X_Bβ − y_B)**

This provides a noisy but unbiased estimate of the true gradient, enabling faster iterations at the cost of convergence smoothness.

## 4. Regularization

### Ridge (L2)

**L_ridge(β) = ‖y − Xβ‖² + α‖β‖₂²**

Closed form: **β̂ = (XᵀX + αI)⁻¹Xᵀy**

Effect: Shrinks all coefficients toward zero. Handles multicollinearity by stabilizing (XᵀX).

### Lasso (L1)

**L_lasso(β) = ‖y − Xβ‖² + α‖β‖₁**

No closed-form solution. Solved via coordinate descent. Produces sparse solutions (some βⱼ = 0), enabling automatic feature selection.

### ElasticNet (L1 + L2)

**L_elastic(β) = ‖y − Xβ‖² + α(ρ‖β‖₁ + (1−ρ)‖β‖₂²)**

Combines the sparsity of Lasso with the stability of Ridge.

## 5. Diagnostic Tests

### Residual Normality

- **Shapiro-Wilk**: Tests H₀: residuals ~ Normal. Sensitive for small samples.
- **Jarque-Bera**: Tests based on skewness and kurtosis. Better for large n.

### Homoscedasticity

- **Breusch-Pagan**: Regresses squared residuals on X. H₀: constant variance. If p < 0.05, heteroscedasticity is present.

### Multicollinearity

- **VIF** (Variance Inflation Factor): VIFⱼ = 1/(1 − Rⱼ²), where Rⱼ² is from regressing feature j on all other features. VIF > 10 indicates problematic collinearity.

### Influence

- **Cook's Distance**: Measures how much all fitted values change when observation i is removed. Threshold: Dᵢ > 4/n.
- **Leverage** (hat values): hᵢᵢ = diagonal of H = X(XᵀX)⁻¹Xᵀ. High leverage ≠ high influence; both must be checked.

## 6. Information Criteria

- **AIC** = n·ln(SS_res/n) + 2(p + 1) — penalizes complexity lightly
- **BIC** = n·ln(SS_res/n) + ln(n)·(p + 1) — penalizes complexity more heavily

Lower values indicate better model fit relative to complexity.

## References

- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*, 2nd ed.
- James, Witten, Hastie, Tibshirani. *An Introduction to Statistical Learning*.
- Greene. *Econometric Analysis*, 8th ed.

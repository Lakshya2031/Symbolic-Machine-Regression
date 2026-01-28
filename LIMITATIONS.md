# Model Limitations

## PyTorch Symbolic Regression - Known Limitations

This document outlines the limitations of the gradient-based symbolic regression model compared to PySR and other symbolic regression approaches.

---

## 1. Fixed Expression Structure

**Limitation:** The model uses a **fixed tree structure** determined at initialization. It cannot dynamically grow or shrink the expression tree during training.

**Impact:**
- May not discover optimal structure if max_depth is too small
- Wastes computation if max_depth is too large
- Cannot adapt complexity during training

**PySR Comparison:** PySR uses genetic programming that can evolve tree structures dynamically.

---

## 2. Softmax Operator Selection (Not Truly Discrete)

**Limitation:** Operators are selected via **softmax-weighted mixtures**, not discrete selection. The model computes a weighted average of ALL operators at each node.

**Impact:**
- Final expression is an approximation until post-training discretization
- May not fully converge to a single operator
- Increased computation (evaluates all operators at each node)
- Discretized expression may perform worse than the continuous model

**Example:**
```
During training: 0.7*sin(x) + 0.2*cos(x) + 0.1*exp(x)
After discretization: sin(x)  ← loses the mixture
```

---

## 3. Local Minima / Gradient Issues

**Limitation:** Gradient descent can get stuck in **local minima**. Discrete symbolic structures create a non-convex optimization landscape.

**Impact:**
- May not find the globally optimal expression
- Sensitive to initialization
- May converge to mathematically equivalent but different expressions

**PySR Comparison:** Evolutionary algorithms explore more broadly and escape local minima through mutations.

---

## 4. Limited Operator Discovery

**Limitation:** Can only discover operators from the **predefined operator set**. Cannot invent new mathematical operations.

**Current Operators:**
- Binary: `+`, `-`, `*`, `/`
- Unary: `sin`, `cos`, `exp`, `log`, `sqrt`, `square`, `neg`, `identity`
- Power: `x^p` (learnable p)

**Cannot discover:**
- `tan`, `arcsin`, `arccos`, `arctan`
- `sinh`, `cosh`, `tanh`
- `abs`, `sign`, `floor`, `ceil`
- Special functions (Bessel, Gamma, etc.)

---

## 5. Coefficient Precision

**Limitation:** Learned constants are **continuous values**, not exact rationals or special constants.

**Impact:**
- May learn `3.14159` instead of exact `π`
- May learn `0.333...` instead of `1/3`
- Post-processing needed to recognize special values

---

## 6. Scaling Issues

**Limitation:** Performance degrades with:
- **Many input variables** (>5 features)
- **Deep expressions** (depth > 4)
- **Complex true equations** (many nested operations)

**Reason:** Number of parameters grows exponentially with depth and features.

---

## 7. Data Requirements

**Limitation:** Requires **sufficient training data** that covers the input domain well.

**Impact:**
- Poor extrapolation outside training range
- May overfit with very few samples (<50)
- Needs data augmentation for sparse datasets

---

## 8. Training Time

**Limitation:** Training is **slow** compared to simple regression models.

**Typical times:**
- Simple equations (depth 2): 30-60 seconds
- Medium equations (depth 3): 2-5 minutes
- Complex equations (depth 4): 5-15 minutes

**PySR Comparison:** PySR parallelizes across populations; this model is sequential.

---

## 9. Expression Interpretability

**Limitation:** Learned expressions may be **mathematically correct but not simplified**.

**Example:**
- True: `x + x`
- Learned: `2*x` or `x * 2` or `(x + x + x) - x`

All are equivalent but less interpretable.

---

## 10. No Compositionality

**Limitation:** Cannot easily **compose learned sub-expressions** or transfer learned patterns.

**Impact:**
- Each new problem trains from scratch
- Cannot reuse discovered building blocks
- No curriculum learning capability

---

## Summary Table

| Aspect | This Model | PySR |
|--------|-----------|------|
| Structure | Fixed | Dynamic |
| Optimization | Gradient descent | Evolutionary + gradient |
| Operator selection | Continuous (softmax) | Discrete |
| Local minima escape | Poor | Good |
| Training speed | Slow | Faster (parallel) |
| Expression simplification | Basic | SymPy integration |
| Compositionality | No | Limited |

---

## When to Use This Model

**Good for:**
- Learning PyTorch-integrated symbolic models
- Differentiable end-to-end pipelines
- Simple to moderate complexity equations
- Educational purposes

**Not ideal for:**
- Production symbolic regression
- Very complex equations
- Large-scale benchmarks
- When exact symbolic recovery is critical

---

## Recommendations

1. **Start simple:** Use `max_depth=2` first, increase if needed
2. **Tune complexity_weight:** Higher values → simpler expressions
3. **Check R² > 0.99:** Lower values suggest structure mismatch
4. **Verify discretized expression:** Compare continuous vs discrete performance
5. **Use more data:** At least 100+ samples for reliable discovery

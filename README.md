# Symbolic Regression with Dynamic Programming Optimization

A pure PyTorch implementation of symbolic regression with DeepChem integration. Unlike the original PySR which uses Julia, this implementation runs entirely in Python/PyTorch for seamless integration with Python ML pipelines.

This project implements an optimized symbolic regression model that:
1. Inherits from DeepChem's `TorchModel` for scientific ML pipelines
2. Uses Dynamic Programming memoization for significant speedup
3. Works with DeepChem datasets (`NumpyDataset`, `DiskDataset`)

## Quick Start

```python
import deepchem as dc
import numpy as np
from symbolic_regression import SymbolicRegressorModel, feynman_to_dataset

# Option 1: Use built-in Feynman equations
dataset, info = feynman_to_dataset('I.6.2', n_samples=1000)
print(f"Target: {info['formula']}")  # '0.5 * m * v^2'

# Option 2: Your own data
X = np.random.randn(1000, 2).astype(np.float32)
y = (X[:, 0]**2 + 2*X[:, 1]).astype(np.float32)
dataset = dc.data.NumpyDataset(X=X, y=y)

# Create model (inherits from dc.models.TorchModel)
model = SymbolicRegressorModel(
    n_features=2,
    max_depth=3,
    complexity_weight=0.01,
    learning_rate=0.01
)

# Train using DeepChem's standard API
loss = model.fit(dataset, nb_epoch=200)

# Get the discovered formula
formula = model.get_formula(var_names=['x', 'y'])
print(f"Discovered: {formula}")
```

## Performance Results

Tested against standard Feynman Benchmark equations (from arXiv:2305.01582v3):

| Test Case | Formula | Baseline Time | Optimized Time | Speedup |
|-----------|---------|---------------|----------------|---------|
| I.6.2 (Kinetic) | E = 0.5mv² | 9.82s | 2.60s | 3.8x |
| I.12.1 (E-Field) | E = q/r² | 16.51s | 2.55s | 6.5x |
| I.29.4 (Wave #) | k = ω/c | 24.56s | 2.44s | 10.1x |

The DP optimization keeps training time consistently around 2.5 seconds even as equations become more complex.

### DP Memoization

The optimization uses several techniques:

| Technique | Description |
|-----------|-------------|
| LRU Cache | Stores subtree evaluation results to avoid redundant computation |
| Hash-based Dedup | Identifies equivalent expressions to prune search space |
| Incremental Eval | Only recomputes changed subtrees for faster parameter updates |
| Subproblem Table | Stores optimal subexpressions for bottom-up construction |

Typical cache hit rate: 85-95% after warmup, resulting in ~6.7x average speedup on Feynman benchmarks.

## Project Structure

```
symbolic_regression/          # Main DeepChem-integrated package
├── __init__.py              # Public API exports
├── src/
│   ├── core/                # Core building blocks
│   │   ├── operators.py     # Differentiable operators (+, -, *, /, sin, cos, etc.)
│   │   ├── nodes.py         # Expression tree nodes
│   │   └── expression.py    # Symbolic expression trees
│   ├── models/
│   │   └── symbolic_regressor.py  # DeepChem TorchModel integration
│   ├── data/
│   │   └── dataset_utils.py       # Feynman equations & dataset utilities
│   ├── optimizers/
│   │   └── dp_cache.py            # Dynamic programming memoization
│   └── benchmarks/
│       └── feynman_benchmark.py   # Benchmark suite
├── tests/
│   └── test_comprehensive.py      # Full test suite
└── README.md

pysr_baseline/               # Original baseline implementation
├── operators.py
├── nodes.py
├── model.py
└── trainer.py

enhancements/                # Optimization enhancements
├── dp_memoization/         # LRU cache & incremental evaluation
└── hybrid_optimization/    # Evolutionary strategies
```

### Key Components

- `symbolic_regression/`: Main package with DeepChem integration (recommended)
  - `src/models/symbolic_regressor.py`: `SymbolicRegressorModel` inherits from `dc.models.TorchModel`
  - `src/optimizers/dp_cache.py`: DP memoization for speedup
- `pysr_baseline/`: Original implementation (operators, tree nodes, basic trainer)
- `enhancements/`: Additional optimizations (LRU cache, evolutionary strategies)

## How It Works

The model is a standard `nn.Module` that builds expression trees using learnable parameters:
- Each node in the tree can be different operators (add, multiply, sin, etc.)
- The model learns which operators to use via softmax weights (differentiable discrete choices)
- Training uses Adam optimizer with a loss that balances accuracy (MSE) and complexity

| File | Description |
|------|-------------|
| `operators.py` | Math operations (binary: +, -, *, /; unary: sin, cos, exp, log, sqrt) |
| `nodes.py` | Building blocks for expression trees (variables, constants, operator nodes) |
| `expression.py` | Symbolic expression trees with multiple structure types |
| `symbolic_regressor.py` | Main model inheriting from `dc.models.TorchModel` |

Data format:
- Works with DeepChem datasets (`NumpyDataset`, `DiskDataset`) or raw PyTorch tensors
- Input shape: `(batch_size, n_features)` 
- Output: predictions as `(batch_size,)` plus a human-readable formula from `.get_formula()`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### DeepChem Integration (Recommended)

```python
from symbolic_regression import SymbolicRegressorModel, feynman_to_dataset
import deepchem as dc

# Load a Feynman benchmark equation
dataset, info = feynman_to_dataset('I.6.2', n_samples=1000)

# Create and train model
model = SymbolicRegressorModel(n_features=2, max_depth=3)
model.fit(dataset, nb_epoch=200)

# Get discovered formula
print(model.get_formula(var_names=['m', 'v']))  # Should discover: 0.5 * m * v^2
```

### Legacy API with DP Optimization

```python
import torch
from enhancements.dp_memoization import OptimizedSymbolicTrainer
from pysr_baseline.model import SymbolicRegressor

model = SymbolicRegressor(n_features=2, max_depth=3)
trainer = OptimizedSymbolicTrainer(model, cache_capacity=1000)
trainer.fit(x_train, y_train, n_epochs=200)
print(f"Result: {model.simplify()}")
```

## Baseline Comparison

### PySR vs This Implementation

| Aspect | Original PySR | This Implementation |
|--------|---------------|---------------------|
| Language | Julia (backend) | Pure PyTorch/Python |
| Integration | Standalone | DeepChem `TorchModel` |
| Optimization | Genetic Programming + Local | Gradient Descent + DP |
| Differentiable | Partially | Fully |
| Dependencies | Julia installation required | Python-only |

### Performance Summary

| Method | Avg Time | Accuracy (R²) |
|--------|----------|---------------|
| GP Baseline | ~30s | 0.85-0.95 |
| Gradient Baseline | 16.96s | 0.92-0.98 |
| DP-Optimized | 2.53s | 0.92-0.98 |

The DP optimization maintains the same accuracy as gradient-based methods while achieving 6.7x speedup through memoization.

## Testing

```bash
# Run all tests
python -m pytest symbolic_regression/tests/ -v

# Run specific test categories
python -m pytest symbolic_regression/tests/ -v -k "TestDeepChemIntegration"
python -m pytest symbolic_regression/tests/ -v -k "TestDP"
python -m pytest symbolic_regression/tests/ -v -k "TestFeynman"

# Run benchmark reproduction
python reproduce_results.py
```

### Test Coverage

| Test Category | Description | Tests |
|---------------|-------------|-------|
| TestCoreOperators | Verifies all math operators | 10 |
| TestOperatorMixtures | Tests softmax-weighted operator selection | 4 |
| TestNodes | Tests expression tree node functionality | 6 |
| TestSymbolicExpression | Tests full expression tree evaluation | 5 |
| TestDeepChemIntegration | Tests TorchModel inheritance & dataset compatibility | 5 |
| TestDPOptimization | Tests DP memoization, caching, and hit rates | 8 |
| TestDPvsBaseline | Compares DP-optimized vs baseline accuracy | 2 |
| TestFeynmanBenchmarks | Tests on Feynman physics equations | 2 |
| TestPyTorchOnly | Verifies no Julia dependencies | 3 |
| TestAccuracy | Tests formula discovery on known equations | 2 |

## DeepChem Integration

The `SymbolicRegressorModel` inherits from DeepChem's `TorchModel`:

| Feature | Status |
|---------|--------|
| Inherits from `TorchModel` | Yes |
| Works with `NumpyDataset` | Yes |
| Works with `DiskDataset` | Yes |
| Standard `.fit()` API | Yes |
| Standard `.predict()` API | Yes |
| GPU support | Yes (via PyTorch CUDA) |
| Checkpointing | Yes |
| Model saving/loading | Yes |

### API Reference

```python
from symbolic_regression import SymbolicRegressorModel

model = SymbolicRegressorModel(
    n_features: int,          # Number of input features
    max_depth: int = 3,       # Expression tree depth
    n_candidates: int = 5,    # Number of candidate expressions
    complexity_weight: float = 0.01,
    learning_rate: float = 0.01,
    batch_size: int = 32
)

# Methods
model.get_formula(var_names=None)      # Get discovered expression as string
model.get_complexity()                 # Get model complexity score
model.get_candidate_info()             # Get all candidate expressions

# Dataset utilities
from symbolic_regression import feynman_to_dataset, create_symbolic_dataset, split_dataset

dataset, info = feynman_to_dataset('I.6.2', n_samples=1000)
train, test = split_dataset(dataset, frac_train=0.8)
```

### Available Feynman Equations

| ID | Formula | Variables |
|----|---------|-----------|
| I.6.2 | E = 0.5mv² | m, v |
| I.12.1 | F = q/r² | q, r |
| I.29.4 | k = ω/c | omega, c |
| I.15.10 | p = m₀v/√(1 - v²/c²) | m0, v, c |
| II.6.15 | E = 3qp_d/(4πεr³) | q, p_d, epsilon, r |

## References

- Cranmer, M., et al. (2023). "Interpretable Machine Learning for Science with PySR". arXiv:2305.01582
- Ramsundar, B., et al. "DeepChem: Democratizing Deep Learning for Drug Discovery"

## License

MIT License - See LICENSE for details.

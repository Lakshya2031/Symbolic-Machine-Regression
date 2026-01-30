# Symbolic Regression Framework

A PyTorch-based symbolic regression system for discovering mathematical formulas from data.


## Overview

Symbolic regression finds mathematical expressions that best fit a given dataset. Unlike traditional regression which fits parameters to a fixed model, symbolic regression searches over the space of possible mathematical expressions to find both the structure and parameters.

This project implements a differentiable approach where operator selection is relaxed using softmax-weighted mixtures, enabling gradient-based optimization of the entire expression tree.


## Project Structure

```
symbolic_regression/
|-- pysr_baseline/              # Core implementation
|   |-- operators.py            # Mathematical operators (unary/binary)
|   |-- nodes.py                # Expression tree node classes
|   |-- model.py                # SymbolicRegressor model
|   |-- trainer.py              # Training loop and optimization
|
|-- enhancements/               # Optimization extensions
|   |-- dp_memoization/         # Dynamic programming for faster evaluation
|   |-- hybrid_optimization/    # Evolutionary + gradient hybrid search
|
|-- benchmarks/                 # Performance evaluation
|-- tests/                      # Unit and integration tests (37 tests)
|-- logs/                       # Development documentation
|
|-- run.py                      # Main entry point
|-- requirements.txt            # Dependencies
```


## Installation

```bash
pip install -r requirements.txt
```

Or simply:
```bash
pip install torch
```


## Usage

Run the demo:
```bash
python run.py
```

Run tests:
```bash
python run.py --test
```

Run benchmarks:
```bash
python run.py --benchmark
```


## Quick Start

```python
import torch
from pysr_baseline.model import SymbolicRegressor
from pysr_baseline.trainer import SymbolicRegressionTrainer

# Generate data for y = x^2 + 2x + 1
x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = x**2 + 2*x + 1

# Create and train model
model = SymbolicRegressor(n_features=1, max_depth=3, n_candidates=5)
trainer = SymbolicRegressionTrainer(model, complexity_weight=0.01)
trainer.fit(x, y, n_epochs=200)

# Get discovered formula
print(model.simplify())
```


## Technical Approach

### Differentiable Expression Trees

Each node in the expression tree maintains softmax weights over available operators. During forward pass, the output is a weighted combination of all operator outputs. During training, these weights converge toward selecting specific operators.

This allows:
- End-to-end gradient flow through the expression
- Smooth optimization landscape
- Standard PyTorch autograd compatibility

### Dynamic Programming Optimization

The DP memoization enhancement caches subtree evaluation results to avoid redundant computation. Key components:

- **ExpressionCache**: LRU cache storing (structure_hash, input_hash) -> output
- **SubproblemTable**: Stores optimal subexpressions per complexity level
- **StructureHasher**: Canonical hashing handling operator commutativity
- **IncrementalEvaluator**: Recomputes only modified subtrees

### Hybrid Optimization

Combines evolutionary search (for structure) with gradient descent (for parameters):

- **Population**: Maintains diverse candidate solutions
- **Mutation**: Parameter, operator, and feature mutations
- **Crossover**: Parameter blending between successful individuals
- **Selection**: Tournament selection with elitism


## Supported Operators

| Type | Operators |
|------|-----------|
| Binary | add, subtract, multiply, divide (protected) |
| Unary | sin, cos, exp, log (protected), sqrt, square, neg, identity |


## Performance Results

Comparison on standard benchmark functions (100 epochs, averaged over 3 runs):

| Method | Average Time | Speedup |
|--------|--------------|---------|
| Baseline | 8.53s | 1.0x |
| DP-Optimized | 2.99s | 2.9x |

The DP optimization achieves significant speedup by caching repeated subtree evaluations.


## Tests

The test suite covers:
- Operator correctness and numerical stability
- Node construction and evaluation
- Model forward pass and gradient flow
- Trainer functionality
- DP caching mechanisms
- Hybrid optimization components
- End-to-end integration

Run with:
```bash
python tests/test_suite.py
```


## References

1. Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv:2305.01582

2. Koza, J.R. (1992). Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press.

3. Bellman, R. (1957). Dynamic Programming. Princeton University Press.


## License

MIT License

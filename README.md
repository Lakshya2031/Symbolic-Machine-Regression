# PyTorch Symbolic Regression

A differentiable symbolic regression implementation in PyTorch that discovers mathematical equations from data. Inspired by [PySR](https://github.com/MilesCranmer/PySR), but uses gradient descent only — no evolutionary algorithms.

## Features

- Automatic equation discovery from data
- 100% gradient-based optimization (backpropagation only)
- Extracts human-readable symbolic expressions
- Native PyTorch integration
- Complexity regularization for simpler equations

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pytorch-symbolic-regression.git
cd pytorch-symbolic-regression

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
python example.py 1
```

## Benchmark Results (Feynman Equations)

Tested on physics equations from the Feynman Lectures:

| Equation | True Formula | R² Score |
|----------|-------------|----------|
| Kinetic Energy | `0.5 * m * v²` | 0.9995 |
| Electric Field | `q / r²` | 0.9999 |
| Wave Number | `ω / c` | 1.0000 |

Mean R²: 0.9998

Run: `python feynman_fast.py`

## Usage

```python
import torch
from model import SymbolicRegressor
from trainer import SymbolicRegressionTrainer

x = torch.rand(500, 1) * 10 - 5
y = x[:, 0]**2 + 2*x[:, 0] + 1

model = SymbolicRegressor(n_features=1, max_depth=3, n_candidates=5)
trainer = SymbolicRegressionTrainer(model, learning_rate=0.02)
trainer.fit(x, y, n_epochs=2000, verbose=1)

print(model.simplify(["x"]))  # ((x * x) + (x + x)) + 1
```

Or edit `main.py` with your data and run `python main.py`.

## Project Structure

```
operators.py      - Operator definitions (+, -, *, /, sin, cos, exp, log)
nodes.py          - Symbolic tree nodes
model.py          - SymbolicRegressor, MultiTermRegressor
trainer.py        - Training loop
simplify.py       - Expression extraction
main.py           - Entry point for custom data
example.py        - Examples on synthetic data
feynman_fast.py   - Benchmark
LIMITATIONS.md    - Known limitations
```

## How It Works

Operators are selected via softmax-weighted mixtures (continuous relaxation of discrete choice). All constants are `nn.Parameter` optimized via gradient descent. Loss = MSE + λ × complexity.

After training, dominant operators are discretized to get a readable formula.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md). Main issues:
- Fixed tree structure
- Softmax operators (not truly discrete)
- Can get stuck in local minima
- Slower than evolutionary approaches

## Comparison with PySR

| Aspect | This Model | PySR |
|--------|-----------|------|
| Optimization | Gradient descent | Evolutionary + gradient |
| Structure | Fixed | Dynamic |
| Speed | Slower | Faster |

## Module Details

### operators.py
Defines the fixed operator space:
- **Binary operators**: `+`, `-`, `*`, `/` (protected division with ε)
- **Unary operators**: `sin`, `cos`, `exp`, `log` (protected), `identity`, `neg`, `square`, `sqrt`
- **Power operator**: `x^p` with learnable exponent p, implemented as `exp(p * log(|x| + ε))`
- **Operator mixtures**: Softmax-weighted combinations for continuous relaxation of discrete operator selection

### nodes.py
Symbolic tree node classes:
- `VariableNode`: Input features x_i
- `ConstantNode`: Learnable constants (`nn.Parameter`)
- `UnaryOpNode`: Unary operator with softmax-weighted mixture
- `BinaryOpNode`: Binary operator with softmax-weighted mixture
- `PowerNode`: Power operation with learnable exponent
- `LinearCombinationNode`: Weighted sum of child nodes
- `WeightedInputNode`: Soft selection of input variables

### model.py
Main symbolic regression models:
- `SymbolicExpression`: Single expression tree with fixed structure
- `SymbolicRegressor`: Multiple candidate expressions with learned combination
- `MultiTermRegressor`: Additive model with explicit term structure

### trainer.py
Training utilities:
- `SymbolicRegressionTrainer`: Full training loop with gradient-based optimization
- Support for Adam and SGD optimizers
- Complexity penalty scheduling (constant, linear, warmup)
- Early stopping and validation

### simplify.py
Post-training tools:
- `ExpressionSimplifier`: Discretize operators, round constants, prune branches
- `extract_expression`: Get structured expression information
- `print_expression_report`: Pretty-print learned expression
- `ExpressionEvaluator`: Evaluate extracted expressions

## Usage

### Basic Example

```python
import torch
from model import SymbolicRegressor
from trainer import SymbolicRegressionTrainer
from simplify import print_expression_report

# Generate synthetic data: f(x) = x^2 + 2*x + 1
x = torch.rand(500, 1) * 10 - 5  # x in [-5, 5]
y = x[:, 0]**2 + 2*x[:, 0] + 1

# Create model
model = SymbolicRegressor(
    n_features=1,
    max_depth=3,
    n_candidates=3,
    complexity_weight=0.001
)

# Train
trainer = SymbolicRegressionTrainer(
    model=model,
    optimizer="adam",
    learning_rate=0.05,
    complexity_weight=0.001
)

results = trainer.fit(
    x, y,
    n_epochs=2000,
    batch_size=64,
    verbose=1
)

# Print learned expression
print_expression_report(model, var_names=["x"])

# Get simplified expression string
simplified = model.simplify(var_names=["x"])
print(f"f(x) = {simplified}")
```

### Run Examples

```bash
# Run all examples
python example.py

# Run specific example (1-6)
python example.py 1  # Simple polynomial
python example.py 2  # Trigonometric function
python example.py 3  # Multivariate function
python example.py 4  # Physical law (Kepler's 3rd)
python example.py 5  # Noisy data
python example.py 6  # Exponential function
```

## Model Architecture

### Symbolic Expression Tree
Expressions are represented as explicit computation graphs:

```
        BinaryOp (softmax over +,-,*,/)
        /           \
    UnaryOp         UnaryOp (softmax over sin,cos,exp,log,...)
       |               |
  WeightedInput   ConstantNode
   (softmax)        (learnable)
```

### Candidate Ensemble
The `SymbolicRegressor` maintains multiple candidate expressions with different structures:
- Binary trees of varying depths
- Mixed trees with unary/binary nodes
- Unary chains

Final output is a softmax-weighted combination of candidates, allowing the model to learn which structure best fits the data.

### Loss Function (PySR-style)
```
Total Loss = MSE + λ × Complexity
```
Where complexity includes:
- Operator complexity (weighted by softmax probabilities)
- Number of active nodes
- Non-zero constant penalties

## Key Design Decisions

1. **Softmax-weighted operator selection**: Provides continuous relaxation of discrete operator choices, enabling gradient-based optimization.

2. **Multiple candidate structures**: Similar to PySR's population of expressions at different complexity levels.

3. **Complexity penalty**: Biases search toward simpler expressions (Occam's razor).

4. **Protected operators**: Division and logarithm use ε-stabilization for numerical safety.

5. **Post-training discretization**: After training, dominant operators are selected and expressions are simplified.

## Comparison with PySR

| Aspect | PySR | This Implementation |
|--------|------|---------------------|
| Optimization | Evolutionary + gradient | Gradient only |
| Operator selection | Discrete mutations | Softmax mixture |
| Expression structure | Variable | Fixed candidates |
| Constant optimization | BFGS/gradient | Gradient (Adam/SGD) |
| Complexity control | Pareto front | Regularization penalty |
| Simplification | SymPy | Custom rules |

## Limitations

- Fixed expression structure (cannot dynamically grow/shrink trees)
- Softmax mixture may not fully converge to single operator
- Complex expressions may require many candidates
- Training may need tuning for different problem types

## References

- [PySR](https://github.com/MilesCranmer/PySR)
- Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR
- Udrescu & Tegmark (2020). AI Feynman

## License

MIT

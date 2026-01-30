"""
PySR Baseline Implementation
----------------------------
A PyTorch-based symbolic regression system that replicates the core 
functionality of PySR (Python Symbolic Regression).

This baseline implements:
    - Expression trees with softmax-weighted operator selection
    - Multiple candidate structures for exploration
    - Gradient-based optimization of continuous parameters
    - Complexity regularization for parsimony

Reference:
    Cranmer, M. (2023). Interpretable Machine Learning for Science 
    with PySR and SymbolicRegression.jl. arXiv:2305.01582
"""

from .operators import (
    OperatorRegistry,
    BinaryOperatorMixture,
    UnaryOperatorMixture,
    add, sub, mul, protected_div,
    sin_op, cos_op, exp_op, protected_log, sqrt_op
)

from .nodes import (
    SymbolicNode,
    VariableNode,
    ConstantNode,
    UnaryOpNode,
    BinaryOpNode,
    WeightedInputNode
)

from .model import (
    SymbolicExpression,
    SymbolicRegressor
)

from .trainer import (
    SymbolicRegressionTrainer
)

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "OperatorRegistry",
    "BinaryOperatorMixture",
    "UnaryOperatorMixture",
    "SymbolicNode",
    "VariableNode",
    "ConstantNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "WeightedInputNode",
    "SymbolicExpression",
    "SymbolicRegressor",
    "SymbolicRegressionTrainer",
]

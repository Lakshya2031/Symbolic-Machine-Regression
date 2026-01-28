"""
PyTorch Symbolic Regression
===========================
A differentiable symbolic regression implementation that mirrors PySR's design philosophy.

Modules:
- operators: Defines the fixed operator space (binary, unary, power operators)
- nodes: Symbolic tree node classes (Variable, Constant, Operator nodes)
- model: Main symbolic regression models (SymbolicRegressor, MultiTermRegressor)
- trainer: Training utilities with gradient-based optimization
- simplify: Post-training expression extraction and simplification

Example Usage:
    from model import SymbolicRegressor
    from trainer import SymbolicRegressionTrainer
    from simplify import print_expression_report
    
    # Create model
    model = SymbolicRegressor(n_features=1, max_depth=3)
    
    # Train
    trainer = SymbolicRegressionTrainer(model, learning_rate=0.01)
    trainer.fit(x_train, y_train, n_epochs=1000)
    
    # Get expression
    print_expression_report(model, var_names=["x"])
"""

__version__ = "1.0.0"
__author__ = "PyTorch Symbolic Regression"

from .operators import (
    OperatorRegistry,
    BinaryOperatorMixture,
    UnaryOperatorMixture,
    LearnablePower
)

from .nodes import (
    SymbolicNode,
    VariableNode,
    ConstantNode,
    UnaryOpNode,
    BinaryOpNode,
    PowerNode,
    LinearCombinationNode,
    WeightedInputNode
)

from .model import (
    SymbolicExpression,
    SymbolicRegressor,
    MultiTermRegressor
)

from .trainer import (
    SymbolicRegressionTrainer,
    train_symbolic_regressor
)

from .simplify import (
    ExpressionSimplifier,
    extract_expression,
    print_expression_report,
    ExpressionEvaluator
)

__all__ = [
    # Operators
    "OperatorRegistry",
    "BinaryOperatorMixture",
    "UnaryOperatorMixture",
    "LearnablePower",
    
    # Nodes
    "SymbolicNode",
    "VariableNode",
    "ConstantNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "PowerNode",
    "LinearCombinationNode",
    "WeightedInputNode",
    
    # Models
    "SymbolicExpression",
    "SymbolicRegressor",
    "MultiTermRegressor",
    
    # Training
    "SymbolicRegressionTrainer",
    "train_symbolic_regressor",
    
    # Simplification
    "ExpressionSimplifier",
    "extract_expression",
    "print_expression_report",
    "ExpressionEvaluator"
]

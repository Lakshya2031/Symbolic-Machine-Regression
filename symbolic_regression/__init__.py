"""
Symbolic Regression with DeepChem Integration

A pure PyTorch implementation of symbolic regression inspired by PySR,
integrated with DeepChem's machine learning infrastructure.

Features:
- Pure PyTorch implementation (no Julia dependency)
- Inherits from DeepChem's TorchModel
- Works with DeepChem's NumpyDataset and DiskDataset
- Dynamic Programming optimization for speedup
- Complexity-regularized loss for interpretable expressions

Usage:
    import deepchem as dc
    from symbolic_regression import SymbolicRegressorModel
    
    dataset = dc.data.NumpyDataset(X=X, y=y)
    model = SymbolicRegressorModel(n_features=2)
    model.fit(dataset, nb_epoch=200)
    print(model.get_formula())
"""

__version__ = "1.0.0"
__author__ = "Lakshya"

# Main DeepChem-integrated model
from .src.models.symbolic_regressor import SymbolicRegressorModel
from .src.models.symbolic_regressor import DPSymbolicRegressorModel

# Dataset utilities
from .src.data.dataset_utils import (
    create_symbolic_dataset,
    feynman_to_dataset,
    split_dataset,
    FEYNMAN_EQUATIONS
)

# Core PyTorch components
from .src.core.operators import OperatorRegistry
from .src.core.nodes import (
    SymbolicNode,
    VariableNode,
    ConstantNode,
    WeightedInputNode,
    UnaryOpNode,
    BinaryOpNode
)
from .src.core.expression import SymbolicExpression

# Optimizations
from .src.optimizers.dp_cache import ExpressionCache, DPOptimizer

# Benchmarks
from .src.benchmarks.feynman_benchmark import FeynmanBenchmark

__all__ = [
    # Main model
    'SymbolicRegressorModel',
    'DPSymbolicRegressorModel',
    
    # Data utilities
    'create_symbolic_dataset',
    'feynman_to_dataset',
    'split_dataset',
    'FEYNMAN_EQUATIONS',
    
    # Core components
    'OperatorRegistry',
    'SymbolicNode',
    'VariableNode',
    'ConstantNode',
    'WeightedInputNode',
    'UnaryOpNode',
    'BinaryOpNode',
    'SymbolicExpression',
    
    # Optimizations
    'ExpressionCache',
    'DPOptimizer',
    
    # Benchmarks
    'FeynmanBenchmark',
]

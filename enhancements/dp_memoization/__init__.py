"""
Dynamic Programming Enhancement for Symbolic Regression
-------------------------------------------------------
Implements memoization and caching strategies to optimize expression
search and evaluation.

Modules:
    - dp_optimizer: Core DP and memoization implementations
"""

from .dp_optimizer import (
    ExpressionCache,
    SubproblemTable,
    StructureHasher,
    MemoizedEvaluator,
    DPSearchOptimizer,
    IncrementalEvaluator,
    OptimizedSymbolicTrainer,
    benchmark_optimization
)

__version__ = "1.0.0"

__all__ = [
    "ExpressionCache",
    "SubproblemTable",
    "StructureHasher",
    "MemoizedEvaluator",
    "DPSearchOptimizer",
    "IncrementalEvaluator",
    "OptimizedSymbolicTrainer",
    "benchmark_optimization"
]

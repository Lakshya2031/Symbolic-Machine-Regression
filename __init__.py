"""
PyTorch Symbolic Regression
===========================
A differentiable symbolic regression implementation that mirrors PySR's design philosophy.

This is the legacy package structure. For the main DeepChem-integrated version,
please use the `symbolic_regression` subpackage:

    from symbolic_regression import SymbolicRegressorModel

Core Modules (Legacy):
- operators: Defines the fixed operator space (binary, unary, power operators)
- nodes: Symbolic tree node classes (Variable, Constant, Operator nodes)
- model: Main symbolic regression models (SymbolicRegressor, MultiTermRegressor)
- trainer: Training utilities with gradient-based optimization
- simplify: Post-training expression extraction and simplification

Enhanced Modules:
- evolutionary: Evolutionary optimization with population-based search
- dynamic_structure: Dynamic tree structure modification during training
- speed_optimizations: Performance optimizations (caching, vectorization, parallel eval)
- hybrid_trainer: Combined evolutionary + gradient optimization
"""

__version__ = "2.0.0"
__author__ = "PyTorch Symbolic Regression"

# Only import if modules exist (for backward compatibility)
try:
    from .operators import (
        OperatorRegistry,
        BinaryOperatorMixture,
        UnaryOperatorMixture,
        LearnablePower
    )
except ImportError:
    pass

try:
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
try:
    from .model import (
        SymbolicExpression,
        SymbolicRegressor,
        MultiTermRegressor
    )
except ImportError:
    pass

try:
    from .trainer import (
        SymbolicRegressionTrainer,
        train_symbolic_regressor
    )
except ImportError:
    pass

try:
    from .simplify import (
        ExpressionSimplifier,
        extract_expression,
        print_expression_report,
        ExpressionEvaluator
    )
except ImportError:
    pass

# Enhanced modules - import conditionally to avoid issues if dependencies missing
try:
    from .evolutionary import (
        Individual,
        EvolutionaryOptimizer,
        IslandModel,
        AdaptiveMutationRate,
        ParameterMutation,
        OperatorMutation,
        ParameterCrossover,
        TournamentSelection
    )
    _HAS_EVOLUTIONARY = True
except ImportError:
    _HAS_EVOLUTIONARY = False

try:
    from .dynamic_structure import (
        DynamicStructureManager,
        AdaptiveComplexityController,
        StructurePruner,
        convert_to_dynamic,
        get_structure_info
    )
    _HAS_DYNAMIC = True
except ImportError:
    _HAS_DYNAMIC = False

try:
    from .speed_optimizations import (
        ExpressionCache,
        VectorizedOperators,
        FastBinaryOperatorMixture,
        FastUnaryOperatorMixture,
        ParallelModelEvaluator,
        PerformanceProfiler,
        benchmark_model
    )
    _HAS_SPEED = True
except ImportError:
    _HAS_SPEED = False

try:
    from .hybrid_trainer import (
        HybridSymbolicTrainer,
        HybridTrainerConfig,
        train_hybrid,
        MultiObjectiveHybridTrainer
    )
    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False

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
    "ExpressionEvaluator",
    
    # Evolutionary (if available)
    "Individual",
    "EvolutionaryOptimizer",
    "IslandModel",
    "AdaptiveMutationRate",
    
    # Dynamic Structure (if available)
    "DynamicStructureManager",
    "AdaptiveComplexityController",
    "convert_to_dynamic",
    "get_structure_info",
    
    # Speed Optimizations (if available)
    "ExpressionCache",
    "VectorizedOperators",
    "FastBinaryOperatorMixture",
    "FastUnaryOperatorMixture",
    "ParallelModelEvaluator",
    "PerformanceProfiler",
    "benchmark_model",
    
    # Hybrid Trainer (if available)
    "HybridSymbolicTrainer",
    "HybridTrainerConfig",
    "train_hybrid",
    "MultiObjectiveHybridTrainer"
]

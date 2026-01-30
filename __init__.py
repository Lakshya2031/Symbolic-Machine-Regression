"""
PyTorch Symbolic Regression
===========================
A differentiable symbolic regression implementation that mirrors PySR's design philosophy.

Core Modules:
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

Example Usage:
    # Standard gradient-based training
    from model import SymbolicRegressor
    from trainer import SymbolicRegressionTrainer
    from simplify import print_expression_report
    
    model = SymbolicRegressor(n_features=1, max_depth=3)
    trainer = SymbolicRegressionTrainer(model, learning_rate=0.01)
    trainer.fit(x_train, y_train, n_epochs=1000)
    print_expression_report(model, var_names=["x"])
    
    # Hybrid evolutionary + gradient training
    from hybrid_trainer import train_hybrid, HybridTrainerConfig
    
    config = HybridTrainerConfig(n_epochs=1000, population_size=20)
    model, results = train_hybrid(x, y, n_features=2, config=config)
"""

__version__ = "2.0.0"
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

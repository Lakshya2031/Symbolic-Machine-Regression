"""
Hybrid Optimization Enhancement
-------------------------------
Combines evolutionary algorithms with gradient-based optimization for
improved symbolic regression performance.

This enhancement implements:
    1. Population-based evolutionary search for structure exploration
    2. Gradient descent for parameter refinement
    3. Adaptive switching between exploration and exploitation phases
"""

from .evolutionary import (
    Individual,
    Population,
    EvolutionaryOptimizer,
    MutationOperator,
    CrossoverOperator
)

from .hybrid_trainer import (
    HybridConfig,
    HybridTrainer,
    AdaptiveScheduler
)

__version__ = "1.0.0"

__all__ = [
    "Individual",
    "Population",
    "EvolutionaryOptimizer",
    "MutationOperator",
    "CrossoverOperator",
    "HybridConfig",
    "HybridTrainer",
    "AdaptiveScheduler"
]

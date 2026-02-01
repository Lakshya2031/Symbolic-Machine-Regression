"""
DeepChem Integration Module
===========================
Provides DeepChem-compatible wrappers for the symbolic regression model.

This module allows the SymbolicRegressor to work seamlessly with 
DeepChem's Dataset classes and follows the TorchModel API pattern.
"""

from .symbolic_regressor_dc import SymbolicRegressorModel
from .dataset_utils import create_symbolic_dataset, feynman_to_dataset

__all__ = [
    'SymbolicRegressorModel',
    'create_symbolic_dataset',
    'feynman_to_dataset'
]

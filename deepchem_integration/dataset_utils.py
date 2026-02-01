"""
Dataset Utilities for Symbolic Regression

Utilities for creating and working with DeepChem datasets.
"""

import numpy as np
from typing import Callable, Optional, List, Tuple, Dict, Any

try:
    import deepchem as dc
    from deepchem.data import NumpyDataset, Dataset
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


def create_symbolic_dataset(
    func: Callable,
    n_samples: int = 1000,
    n_features: int = 2,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    var_names: Optional[List[str]] = None
) -> 'NumpyDataset':
    """
    Create a DeepChem dataset from a mathematical function.
    
    This is useful for testing symbolic regression on known functions
    to verify that the model can recover the true expression.
    
    Parameters
    ----------
    func : callable
        Function that takes a numpy array of shape (n_samples, n_features)
        and returns an array of shape (n_samples,)
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of input features
    x_range : tuple of float
        Range (min, max) for input values
    noise_level : float
        Standard deviation of Gaussian noise to add to outputs
    seed : int, optional
        Random seed for reproducibility
    var_names : list of str, optional
        Names for the input variables (stored as metadata)
        
    Returns
    -------
    NumpyDataset
        DeepChem dataset with the generated data
        
    Examples
    --------
    >>> # Create dataset for y = x0^2 + sin(x1)
    >>> def target_func(X):
    ...     return X[:, 0]**2 + np.sin(X[:, 1])
    >>> dataset = create_symbolic_dataset(target_func, n_samples=1000, n_features=2)
    """
    if not DEEPCHEM_AVAILABLE:
        raise ImportError("DeepChem is required. Install with: pip install deepchem")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random input data
    X = np.random.uniform(x_range[0], x_range[1], size=(n_samples, n_features))
    
    # Compute target values
    y = func(X)
    
    # Add noise if specified
    if noise_level > 0:
        y = y + np.random.normal(0, noise_level, size=y.shape)
    
    # Ensure y is float32
    y = y.astype(np.float32)
    X = X.astype(np.float32)
    
    # Create dataset
    dataset = NumpyDataset(X=X, y=y)
    
    return dataset


def feynman_to_dataset(
    equation_id: str,
    n_samples: int = 1000,
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> Tuple['NumpyDataset', Dict[str, Any]]:
    """
    Create a DeepChem dataset from a Feynman equation.
    
    The Feynman equations are standard benchmarks for symbolic regression,
    taken from physics formulas in the Feynman Lectures.
    
    Parameters
    ----------
    equation_id : str
        Identifier for the equation (e.g., 'I.6.2', 'I.12.1', 'I.29.4')
    n_samples : int
        Number of samples to generate
    noise_level : float
        Standard deviation of Gaussian noise
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (NumpyDataset, dict with equation info)
        
    Examples
    --------
    >>> dataset, info = feynman_to_dataset('I.6.2', n_samples=1000)
    >>> print(info['formula'])  # '0.5 * m * v^2'
    >>> print(info['var_names'])  # ['m', 'v']
    """
    if not DEEPCHEM_AVAILABLE:
        raise ImportError("DeepChem is required. Install with: pip install deepchem")
    
    # Define Feynman equations
    FEYNMAN_EQUATIONS = {
        'I.6.2': {
            'formula': '0.5 * m * v^2',
            'description': 'Kinetic Energy',
            'var_names': ['m', 'v'],
            'func': lambda X: 0.5 * X[:, 0] * X[:, 1]**2,
            'x_ranges': [(0.1, 10.0), (0.1, 10.0)]
        },
        'I.12.1': {
            'formula': 'q / r^2',
            'description': 'Electric Field (simplified)',
            'var_names': ['q', 'r'],
            'func': lambda X: X[:, 0] / (X[:, 1]**2 + 1e-6),
            'x_ranges': [(0.1, 10.0), (0.5, 10.0)]
        },
        'I.29.4': {
            'formula': 'omega / c',
            'description': 'Wave Number',
            'var_names': ['omega', 'c'],
            'func': lambda X: X[:, 0] / (X[:, 1] + 1e-6),
            'x_ranges': [(0.1, 10.0), (0.1, 10.0)]
        },
        'I.15.3x': {
            'formula': 'x - v*t',
            'description': 'Position (Galilean)',
            'var_names': ['x', 'v', 't'],
            'func': lambda X: X[:, 0] - X[:, 1] * X[:, 2],
            'x_ranges': [(-10.0, 10.0), (-5.0, 5.0), (0.0, 10.0)]
        },
        'I.18.4': {
            'formula': 'm1*r1 + m2*r2 / (m1 + m2)',
            'description': 'Center of Mass',
            'var_names': ['m1', 'r1', 'm2', 'r2'],
            'func': lambda X: (X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3]) / (X[:, 0] + X[:, 2] + 1e-6),
            'x_ranges': [(0.1, 10.0), (-5.0, 5.0), (0.1, 10.0), (-5.0, 5.0)]
        },
        'II.6.15a': {
            'formula': 'epsilon * E^2 / 2',
            'description': 'Electric Field Energy Density',
            'var_names': ['epsilon', 'E'],
            'func': lambda X: X[:, 0] * X[:, 1]**2 / 2,
            'x_ranges': [(0.1, 10.0), (-5.0, 5.0)]
        }
    }
    
    if equation_id not in FEYNMAN_EQUATIONS:
        available = list(FEYNMAN_EQUATIONS.keys())
        raise ValueError(f"Unknown equation: {equation_id}. Available: {available}")
    
    eq_info = FEYNMAN_EQUATIONS[equation_id]
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate data
    n_features = len(eq_info['var_names'])
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for i, (low, high) in enumerate(eq_info['x_ranges']):
        X[:, i] = np.random.uniform(low, high, n_samples)
    
    y = eq_info['func'](X).astype(np.float32)
    
    if noise_level > 0:
        y = y + np.random.normal(0, noise_level, size=y.shape).astype(np.float32)
    
    dataset = NumpyDataset(X=X, y=y)
    
    info = {
        'equation_id': equation_id,
        'formula': eq_info['formula'],
        'description': eq_info['description'],
        'var_names': eq_info['var_names'],
        'n_features': n_features
    }
    
    return dataset, info


def split_dataset(
    dataset: 'Dataset',
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    seed: Optional[int] = None
) -> Tuple['NumpyDataset', 'NumpyDataset', 'NumpyDataset']:
    """
    Split a dataset into train/validation/test sets.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to split
    train_frac : float
        Fraction for training set
    valid_frac : float
        Fraction for validation set
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple of NumpyDataset
        (train_dataset, valid_dataset, test_dataset)
    """
    if not DEEPCHEM_AVAILABLE:
        raise ImportError("DeepChem is required")
    
    if seed is not None:
        np.random.seed(seed)
    
    n = len(dataset)
    indices = np.random.permutation(n)
    
    n_train = int(train_frac * n)
    n_valid = int(valid_frac * n)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    
    train_dataset = NumpyDataset(
        X=dataset.X[train_idx],
        y=dataset.y[train_idx],
        w=dataset.w[train_idx]
    )
    
    valid_dataset = NumpyDataset(
        X=dataset.X[valid_idx],
        y=dataset.y[valid_idx],
        w=dataset.w[valid_idx]
    )
    
    test_dataset = NumpyDataset(
        X=dataset.X[test_idx],
        y=dataset.y[test_idx],
        w=dataset.w[test_idx]
    )
    
    return train_dataset, valid_dataset, test_dataset

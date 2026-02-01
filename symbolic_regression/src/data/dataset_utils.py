"""
Dataset Utilities for Symbolic Regression

Utilities for creating and working with DeepChem datasets
for symbolic regression tasks, including Feynman benchmark equations.
"""

import numpy as np
from typing import Callable, Optional, List, Tuple, Dict, Any

try:
    import deepchem as dc
    from deepchem.data import NumpyDataset, Dataset
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


# Feynman Benchmark Equations

FEYNMAN_EQUATIONS = {
    # Volume I - Mechanics
    'I.6.2': {
        'formula': '0.5 * m * v^2',
        'latex': r'E = \frac{1}{2}mv^2',
        'description': 'Kinetic Energy',
        'var_names': ['m', 'v'],
        'func': lambda X: 0.5 * X[:, 0] * X[:, 1]**2,
        'x_ranges': [(0.1, 10.0), (0.1, 10.0)],
        'difficulty': 'easy'
    },
    'I.6.2a': {
        'formula': 'E / (0.5 * m)',
        'latex': r'v^2 = \frac{2E}{m}',
        'description': 'Velocity from Kinetic Energy',
        'var_names': ['E', 'm'],
        'func': lambda X: X[:, 0] / (0.5 * X[:, 1] + 1e-6),
        'x_ranges': [(0.1, 10.0), (0.1, 10.0)],
        'difficulty': 'easy'
    },
    'I.12.1': {
        'formula': 'q / r^2',
        'latex': r'E = \frac{q}{r^2}',
        'description': 'Electric Field (simplified)',
        'var_names': ['q', 'r'],
        'func': lambda X: X[:, 0] / (X[:, 1]**2 + 1e-6),
        'x_ranges': [(0.1, 10.0), (0.5, 10.0)],
        'difficulty': 'medium'
    },
    'I.15.3x': {
        'formula': 'x - v*t',
        'latex': r"x' = x - vt",
        'description': 'Galilean Transformation',
        'var_names': ['x', 'v', 't'],
        'func': lambda X: X[:, 0] - X[:, 1] * X[:, 2],
        'x_ranges': [(-10.0, 10.0), (-5.0, 5.0), (0.0, 10.0)],
        'difficulty': 'easy'
    },
    'I.18.4': {
        'formula': '(m1*r1 + m2*r2) / (m1 + m2)',
        'latex': r'r_{cm} = \frac{m_1 r_1 + m_2 r_2}{m_1 + m_2}',
        'description': 'Center of Mass',
        'var_names': ['m1', 'r1', 'm2', 'r2'],
        'func': lambda X: (X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3]) / (X[:, 0] + X[:, 2] + 1e-6),
        'x_ranges': [(0.1, 10.0), (-5.0, 5.0), (0.1, 10.0), (-5.0, 5.0)],
        'difficulty': 'hard'
    },
    'I.29.4': {
        'formula': 'omega / c',
        'latex': r'k = \frac{\omega}{c}',
        'description': 'Wave Number',
        'var_names': ['omega', 'c'],
        'func': lambda X: X[:, 0] / (X[:, 1] + 1e-6),
        'x_ranges': [(0.1, 10.0), (0.1, 10.0)],
        'difficulty': 'easy'
    },
    'I.32.5': {
        'formula': 'q^2 * a^2 / (6 * pi * eps * c^3)',
        'latex': r'P = \frac{q^2 a^2}{6\pi\epsilon c^3}',
        'description': 'Larmor Formula (simplified)',
        'var_names': ['q', 'a', 'eps', 'c'],
        'func': lambda X: X[:, 0]**2 * X[:, 1]**2 / (6 * np.pi * X[:, 2] * X[:, 3]**3 + 1e-6),
        'x_ranges': [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.5, 5.0)],
        'difficulty': 'hard'
    },
    
    # Volume II - Electromagnetism  
    'II.6.15a': {
        'formula': 'epsilon * E^2 / 2',
        'latex': r'u = \frac{\epsilon E^2}{2}',
        'description': 'Electric Field Energy Density',
        'var_names': ['epsilon', 'E'],
        'func': lambda X: X[:, 0] * X[:, 1]**2 / 2,
        'x_ranges': [(0.1, 10.0), (-5.0, 5.0)],
        'difficulty': 'easy'
    },
    'II.11.3': {
        'formula': 'q * E_f / m',
        'latex': r'a = \frac{qE}{m}',
        'description': 'Acceleration in Electric Field',
        'var_names': ['q', 'E_f', 'm'],
        'func': lambda X: X[:, 0] * X[:, 1] / (X[:, 2] + 1e-6),
        'x_ranges': [(0.1, 10.0), (-5.0, 5.0), (0.1, 10.0)],
        'difficulty': 'medium'
    },
    'II.35.21': {
        'formula': 'n_0 * exp(-m * g * x / (k * T))',
        'latex': r'n = n_0 e^{-mgx/kT}',
        'description': 'Barometric Formula',
        'var_names': ['n_0', 'm', 'g', 'x', 'k', 'T'],
        'func': lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 2] * X[:, 3] / (X[:, 4] * X[:, 5] + 1e-6)),
        'x_ranges': [(0.1, 5.0), (0.1, 2.0), (0.1, 2.0), (0.0, 2.0), (0.1, 2.0), (0.5, 5.0)],
        'difficulty': 'hard'
    },
    
    # Trigonometric functions
    'trig.1': {
        'formula': 'sin(x)',
        'latex': r'y = \sin(x)',
        'description': 'Sine Function',
        'var_names': ['x'],
        'func': lambda X: np.sin(X[:, 0]),
        'x_ranges': [(-3.14, 3.14)],
        'difficulty': 'easy'
    },
    'trig.2': {
        'formula': 'sin(x) + cos(y)',
        'latex': r'z = \sin(x) + \cos(y)',
        'description': 'Sine + Cosine',
        'var_names': ['x', 'y'],
        'func': lambda X: np.sin(X[:, 0]) + np.cos(X[:, 1]),
        'x_ranges': [(-3.14, 3.14), (-3.14, 3.14)],
        'difficulty': 'medium'
    },
}


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
        Names for the input variables
        
    Returns
    -------
    NumpyDataset
        DeepChem dataset with the generated data
        
    Examples
    --------
    >>> def target_func(X):
    ...     return X[:, 0]**2 + np.sin(X[:, 1])
    >>> dataset = create_symbolic_dataset(target_func, n_samples=1000, n_features=2)
    """
    if not DEEPCHEM_AVAILABLE:
        raise ImportError("DeepChem required. Install with: pip install deepchem")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random input data
    X = np.random.uniform(x_range[0], x_range[1], size=(n_samples, n_features))
    
    # Compute target values
    y = func(X)
    
    # Add noise if specified
    if noise_level > 0:
        y = y + np.random.normal(0, noise_level, size=y.shape)
    
    # Ensure correct dtypes
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    return NumpyDataset(X=X, y=y)


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
        raise ImportError("DeepChem required. Install with: pip install deepchem")
    
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
        'latex': eq_info.get('latex', ''),
        'description': eq_info['description'],
        'var_names': eq_info['var_names'],
        'n_features': n_features,
        'difficulty': eq_info.get('difficulty', 'unknown')
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
        Fraction for validation set (rest goes to test)
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple of NumpyDataset
        (train_dataset, valid_dataset, test_dataset)
    """
    if not DEEPCHEM_AVAILABLE:
        raise ImportError("DeepChem required")
    
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


def list_feynman_equations() -> List[Dict[str, Any]]:
    """
    List all available Feynman equations with their metadata.
    
    Returns
    -------
    list of dict
        List of equation information dictionaries
    """
    equations = []
    for eq_id, info in FEYNMAN_EQUATIONS.items():
        equations.append({
            'id': eq_id,
            'formula': info['formula'],
            'description': info['description'],
            'n_variables': len(info['var_names']),
            'difficulty': info.get('difficulty', 'unknown')
        })
    return equations

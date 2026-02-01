"""
DeepChem-Compatible Symbolic Regression Model

Provides a symbolic regression model that inherits from DeepChem's TorchModel.

Usage:
    import deepchem as dc
    from deepchem_integration import SymbolicRegressorModel
    
    X = np.random.randn(1000, 2)
>>> y = 0.5 * X[:, 0] * X[:, 1]**2  # Target: 0.5 * x0 * x1^2
>>> dataset = dc.data.NumpyDataset(X=X, y=y)
>>> 
>>> # Create and train model
>>> model = SymbolicRegressorModel(n_features=2, max_depth=3)
>>> model.fit(dataset, nb_epoch=100)
>>> 
>>> # Get predictions and discovered formula
>>> predictions = model.predict(dataset)
>>> formula = model.get_formula()
>>> print(f"Discovered: {formula}")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union, Iterable

try:
    import deepchem as dc
    from deepchem.models.torch_models import TorchModel
    from deepchem.models.losses import Loss, L2Loss
    from deepchem.data import Dataset, NumpyDataset
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False
    TorchModel = nn.Module  # Fallback for standalone use

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysr_baseline.model import SymbolicRegressor as BaseSymbolicRegressor
from pysr_baseline.nodes import (
    SymbolicNode, WeightedInputNode, UnaryOpNode, BinaryOpNode
)


class SymbolicRegressionLoss(Loss):
    """
    Custom loss for symbolic regression that combines MSE with complexity penalty.
    
    Loss = MSE(y_pred, y_true) + λ * complexity
    
    This encourages the model to find simple expressions that fit the data well.
    """
    
    def __init__(self, complexity_weight: float = 0.01):
        """
        Parameters
        ----------
        complexity_weight : float
            Weight λ for the complexity penalty term
        """
        super().__init__()
        self.complexity_weight = complexity_weight
        self._model = None  # Will be set by SymbolicRegressorModel
    
    def _compute_tf_loss(self, output, labels):
        """TensorFlow loss computation (not implemented)."""
        raise NotImplementedError("TensorFlow not supported")
    
    def _create_pytorch_loss(self):
        """Create PyTorch loss function."""
        def loss_fn(outputs, labels, weights):
            # outputs is a list, get the prediction
            if isinstance(outputs, list):
                y_pred = outputs[0]
            else:
                y_pred = outputs
            
            # Squeeze dimensions if needed
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze(-1)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            
            # MSE loss
            mse = torch.mean((y_pred - labels) ** 2)
            
            # Add complexity penalty if model is available
            if self._model is not None and hasattr(self._model, 'get_complexity'):
                complexity = self._model.get_complexity()
                total_loss = mse + self.complexity_weight * complexity
            else:
                total_loss = mse
            
            return total_loss
        
        return loss_fn


class SymbolicRegressorModule(nn.Module):
    """
    PyTorch Module wrapper for symbolic regression.
    
    This wraps the base SymbolicRegressor to ensure compatibility
    with DeepChem's TorchModel expectations.
    """
    
    def __init__(
        self,
        n_features: int,
        max_depth: int = 3,
        n_candidates: int = 5,
        structures: Optional[List[str]] = None,
        complexity_weight: float = 0.01
    ):
        super().__init__()
        self.n_features = n_features
        self.max_depth = max_depth
        self.n_candidates = n_candidates
        self.complexity_weight = complexity_weight
        
        if structures is None:
            structures = ["binary_tree", "mixed", "unary_chain"]
        self.structures = structures
        
        # Build the symbolic regression components
        self._build_model()
    
    def _build_model(self):
        """Build the symbolic expression candidates."""
        from pysr_baseline.model import SymbolicExpression
        
        self.candidates = nn.ModuleList()
        for i in range(self.n_candidates):
            structure = self.structures[i % len(self.structures)]
            depth = max(1, self.max_depth - i // len(self.structures))
            expr = SymbolicExpression(self.n_features, depth, structure)
            self.candidates.append(expr)
        
        # Learnable weights for candidate selection
        self.candidate_logits = nn.Parameter(torch.zeros(self.n_candidates))
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the symbolic regression model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features)
            
        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        # Ensure input is float
        if x.dtype != torch.float32:
            x = x.float()
        
        # Get candidate weights via softmax
        weights = torch.softmax(self.candidate_logits, dim=0)
        
        # Evaluate all candidates
        outputs = torch.stack([cand(x) for cand in self.candidates], dim=1)
        
        # Weighted combination
        result = torch.sum(outputs * weights, dim=1)
        
        # Apply output scaling
        result = result * self.output_scale + self.output_bias
        
        # Return with shape (batch_size, 1) for DeepChem compatibility
        return result.unsqueeze(-1)
    
    def get_complexity(self) -> torch.Tensor:
        """Get expected complexity based on candidate weights."""
        weights = torch.softmax(self.candidate_logits, dim=0)
        complexities = torch.stack([cand.get_complexity() for cand in self.candidates])
        return torch.sum(weights * complexities)
    
    def get_candidate_weights(self) -> torch.Tensor:
        """Return softmax weights for candidates."""
        return torch.softmax(self.candidate_logits, dim=0)
    
    def get_dominant_candidate(self) -> Tuple[int, nn.Module]:
        """Return the dominant candidate (highest weight)."""
        weights = self.get_candidate_weights()
        idx = torch.argmax(weights).item()
        return idx, self.candidates[idx]
    
    def simplify(self, var_names: Optional[List[str]] = None) -> str:
        """
        Get simplified expression using dominant candidate.
        
        Parameters
        ----------
        var_names : list of str, optional
            Variable names to use in the expression
            
        Returns
        -------
        str
            Human-readable mathematical expression
        """
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        
        idx, candidate = self.get_dominant_candidate()
        base_expr = candidate.simplify(var_names)
        
        scale = self.output_scale.item()
        bias = self.output_bias.item()
        
        if abs(scale - 1.0) > 0.01:
            base_expr = f"{scale:.4f} * ({base_expr})"
        if abs(bias) > 0.01:
            base_expr = f"({base_expr}) + {bias:.4f}"
        
        return base_expr


if DEEPCHEM_AVAILABLE:
    class SymbolicRegressorModel(TorchModel):
        """
        DeepChem-compatible Symbolic Regression Model.
        
        This model inherits from DeepChem's TorchModel and provides symbolic
        regression capabilities that integrate with DeepChem's dataset and
        evaluation infrastructure.
        
        The model learns mathematical expressions from data by maintaining
        multiple candidate expression trees and optimizing them using 
        gradient descent with complexity regularization.
        
        Parameters
        ----------
        n_features : int
            Number of input features
        max_depth : int, default 3
            Maximum depth of expression trees
        n_candidates : int, default 5
            Number of candidate expressions to maintain
        structures : list of str, optional
            Types of tree structures to use. Options: 'binary_tree', 
            'mixed', 'unary_chain'
        complexity_weight : float, default 0.01
            Weight for complexity penalty in loss function
        learning_rate : float, default 0.01
            Learning rate for optimization
        batch_size : int, default 32
            Batch size for training
        model_dir : str, optional
            Directory to save model checkpoints
        device : torch.device, optional
            Device to run computations on
            
        Examples
        --------
        >>> import deepchem as dc
        >>> import numpy as np
        >>> 
        >>> # Generate synthetic data: y = x0^2 + 2*x1
        >>> X = np.random.randn(1000, 2)
        >>> y = X[:, 0]**2 + 2*X[:, 1]
        >>> dataset = dc.data.NumpyDataset(X=X, y=y)
        >>> 
        >>> # Create and train model
        >>> model = SymbolicRegressorModel(n_features=2, max_depth=4)
        >>> loss = model.fit(dataset, nb_epoch=200)
        >>> 
        >>> # Get the discovered formula
        >>> print(model.get_formula())
        
        Notes
        -----
        This implementation follows the PySR approach of using differentiable
        expression trees with softmax-weighted operator selection. The key
        difference from traditional symbolic regression is that operators are
        chosen via continuous optimization rather than discrete search.
        
        References
        ----------
        .. [1] Cranmer, M. (2023). Interpretable Machine Learning for Science 
               with PySR and SymbolicRegression.jl. arXiv:2305.01582
        """
        
        def __init__(
            self,
            n_features: int,
            max_depth: int = 3,
            n_candidates: int = 5,
            structures: Optional[List[str]] = None,
            complexity_weight: float = 0.01,
            learning_rate: float = 0.01,
            batch_size: int = 32,
            model_dir: Optional[str] = None,
            device: Optional[torch.device] = None,
            **kwargs
        ):
            # Create the PyTorch module
            self.n_features = n_features
            self.max_depth = max_depth
            self.n_candidates = n_candidates
            self.complexity_weight = complexity_weight
            
            pytorch_model = SymbolicRegressorModule(
                n_features=n_features,
                max_depth=max_depth,
                n_candidates=n_candidates,
                structures=structures,
                complexity_weight=complexity_weight
            )
            
            # Create custom loss
            loss = SymbolicRegressionLoss(complexity_weight=complexity_weight)
            loss._model = pytorch_model
            
            # Initialize TorchModel
            super().__init__(
                model=pytorch_model,
                loss=loss,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_dir=model_dir,
                device=device,
                **kwargs
            )
        
        def get_formula(self, var_names: Optional[List[str]] = None) -> str:
            """
            Get the discovered mathematical formula.
            
            Parameters
            ----------
            var_names : list of str, optional
                Names for input variables. Defaults to x0, x1, ...
                
            Returns
            -------
            str
                Human-readable mathematical expression
            """
            return self.model.simplify(var_names)
        
        def get_complexity(self) -> float:
            """
            Get the complexity of the current model.
            
            Returns
            -------
            float
                Complexity score (lower is simpler)
            """
            with torch.no_grad():
                return self.model.get_complexity().item()
        
        def get_candidate_info(self) -> Dict[str, Any]:
            """
            Get information about all candidate expressions.
            
            Returns
            -------
            dict
                Dictionary containing weights and expressions for each candidate
            """
            weights = self.model.get_candidate_weights().detach().cpu().numpy()
            info = {
                'n_candidates': self.n_candidates,
                'candidates': []
            }
            
            for i, (w, cand) in enumerate(zip(weights, self.model.candidates)):
                info['candidates'].append({
                    'index': i,
                    'weight': float(w),
                    'complexity': cand.get_complexity().item(),
                    'expression': cand.simplify()
                })
            
            return info
        
        def fit_with_early_stopping(
            self,
            dataset: Dataset,
            validation_dataset: Optional[Dataset] = None,
            nb_epoch: int = 100,
            patience: int = 20,
            min_delta: float = 1e-6,
            **kwargs
        ) -> List[float]:
            """
            Train with early stopping based on validation loss.
            
            Parameters
            ----------
            dataset : Dataset
                Training dataset
            validation_dataset : Dataset, optional
                Validation dataset. If None, uses 10% of training data.
            nb_epoch : int
                Maximum number of epochs
            patience : int
                Number of epochs without improvement before stopping
            min_delta : float
                Minimum change to qualify as an improvement
                
            Returns
            -------
            list of float
                Training loss history
            """
            if validation_dataset is None:
                # Split dataset
                n = len(dataset)
                n_val = max(1, int(0.1 * n))
                indices = np.random.permutation(n)
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]
                
                train_dataset = NumpyDataset(
                    X=dataset.X[train_indices],
                    y=dataset.y[train_indices],
                    w=dataset.w[train_indices]
                )
                validation_dataset = NumpyDataset(
                    X=dataset.X[val_indices],
                    y=dataset.y[val_indices],
                    w=dataset.w[val_indices]
                )
            else:
                train_dataset = dataset
            
            best_loss = float('inf')
            patience_counter = 0
            all_losses = []
            best_state = None
            
            for epoch in range(nb_epoch):
                # Train for one epoch
                loss = self.fit(train_dataset, nb_epoch=1, **kwargs)
                all_losses.append(loss)
                
                # Evaluate on validation set
                val_pred = self.predict(validation_dataset)
                val_loss = np.mean((val_pred.squeeze() - validation_dataset.y.squeeze()) ** 2)
                
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Restore best model
            if best_state is not None:
                self.model.load_state_dict(best_state)
            
            return all_losses

else:
    # Fallback when DeepChem is not available
    class SymbolicRegressorModel(nn.Module):
        """Standalone version when DeepChem is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DeepChem is required for SymbolicRegressorModel. "
                "Install it with: pip install deepchem"
            )

"""
Training Module
---------------
Gradient-based trainer for symbolic regression models.
Implements standard training loop with complexity regularization
and early stopping based on validation loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List, Tuple
import time


class SymbolicRegressionTrainer:
    """
    Trainer for symbolic regression models using gradient descent.
    
    Features:
        - Mini-batch training with complexity regularization
        - Learning rate scheduling
        - Early stopping on validation loss
        - Training history logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        complexity_weight: float = 0.01,
        optimizer_type: str = "adam"
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.complexity_weight = complexity_weight
        
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20
        )
        self.history: List[Dict[str, float]] = []
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 50,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train the model on the provided data.
        
        Args:
            x: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, 1)
            n_epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction of data for validation
            early_stopping_patience: Stop if no improvement for this many epochs
            verbose: Print training progress
            
        Returns:
            Training history as list of metric dictionaries
        """
        if y.dim() == 2:
            y = y.squeeze(-1)
        
        n_samples = x.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_metrics = self._train_epoch(train_loader, epoch, n_epochs)
            
            with torch.no_grad():
                val_pred = self.model(x_val)
                val_mse = torch.mean((val_pred - y_val) ** 2).item()
                val_complexity = self.model.get_complexity().item()
                val_loss = val_mse + self.complexity_weight * val_complexity
            
            epoch_metrics.update({
                "val_mse": val_mse,
                "val_loss": val_loss,
                "val_complexity": val_complexity
            })
            self.history.append(epoch_metrics)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train MSE: {epoch_metrics['mse']:.6f} | "
                      f"Val MSE: {val_mse:.6f} | "
                      f"Complexity: {val_complexity:.2f} | "
                      f"Time: {elapsed:.1f}s")
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_complexity = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            loss, components = self.model.compute_loss(
                batch_x, batch_y, self.complexity_weight
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += components["mse"].item()
            total_complexity += components["complexity"].item()
            n_batches += 1
        
        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "complexity": total_complexity / n_batches,
            "epoch": epoch
        }
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        if y.dim() == 2:
            y = y.squeeze(-1)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            mse = torch.mean((y_pred - y) ** 2).item()
            
            y_mean = torch.mean(y)
            ss_tot = torch.sum((y - y_mean) ** 2)
            ss_res = torch.sum((y - y_pred) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            complexity = self.model.get_complexity().item()
        
        return {
            "mse": mse,
            "rmse": mse ** 0.5,
            "r2": r2.item(),
            "complexity": complexity
        }

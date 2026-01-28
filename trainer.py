"""
Training Module
===============
Provides training utilities for the symbolic regression model:
- Gradient-based optimization (SGD, Adam)
- Learning rate scheduling
- Training loop with logging
- Early stopping based on loss plateau

All parameter updates are performed exclusively through backpropagation.
No evolutionary or heuristic methods are used.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple, Callable, Any
import time
from model import SymbolicRegressor, MultiTermRegressor


class SymbolicRegressionTrainer:
    """
    Trainer class for symbolic regression models.
    
    Uses standard gradient descent optimization to update:
    - Constant parameters
    - Operator selection weights (via softmax)
    - Linear combination weights
    - Power exponents
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "adam",
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        complexity_weight: float = 0.01,
        complexity_schedule: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model: SymbolicRegressor or MultiTermRegressor
            optimizer: "adam" or "sgd"
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            complexity_weight: Initial complexity penalty weight (λ)
            complexity_schedule: Schedule for complexity weight
                - None: constant
                - "linear": linearly increase
                - "warmup": start low, then increase
            device: Device to train on ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.device = device
        self.initial_lr = learning_rate
        self.complexity_weight = complexity_weight
        self.complexity_schedule = complexity_schedule
        
        # Setup optimizer
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "mse": [],
            "complexity": [],
            "complexity_weight": [],
            "lr": []
        }
    
    def get_complexity_weight(self, epoch: int, n_epochs: int) -> float:
        """Get complexity weight based on schedule."""
        if self.complexity_schedule is None:
            return self.complexity_weight
        elif self.complexity_schedule == "linear":
            # Linearly increase from 0 to complexity_weight
            return self.complexity_weight * (epoch + 1) / n_epochs
        elif self.complexity_schedule == "warmup":
            # Start at 10% for first 20% of training, then linear increase
            warmup_epochs = n_epochs // 5
            if epoch < warmup_epochs:
                return self.complexity_weight * 0.1
            else:
                progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
                return self.complexity_weight * (0.1 + 0.9 * progress)
        else:
            return self.complexity_weight
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        n_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with (x, y) batches
            epoch: Current epoch number
            n_epochs: Total number of epochs
            
        Returns:
            Dictionary of average loss components
        """
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_complexity = 0.0
        n_batches = 0
        
        # Get current complexity weight
        curr_complexity_weight = self.get_complexity_weight(epoch, n_epochs)
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, components = self.model.compute_loss(
                batch_x, batch_y, 
                complexity_weight=curr_complexity_weight
            )
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_mse += components["mse"].item()
            total_complexity += components["complexity"].item()
            n_batches += 1
        
        # Average metrics
        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_complexity = total_complexity / n_batches
        
        # Update history
        self.history["loss"].append(avg_loss)
        self.history["mse"].append(avg_mse)
        self.history["complexity"].append(avg_complexity)
        self.history["complexity_weight"].append(curr_complexity_weight)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
        
        return {
            "loss": avg_loss,
            "mse": avg_mse,
            "complexity": avg_complexity,
            "complexity_weight": curr_complexity_weight
        }
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(x)
            mse = torch.mean((y_pred - y) ** 2).item()
            mae = torch.mean(torch.abs(y_pred - y)).item()
            
            # R-squared
            ss_res = torch.sum((y - y_pred) ** 2)
            ss_tot = torch.sum((y - torch.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8)).item()
            
            complexity = self.model.get_complexity().item()
        
        return {
            "mse": mse,
            "rmse": mse ** 0.5,
            "mae": mae,
            "r2": r2,
            "complexity": complexity
        }
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 1000,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 100,
        verbose: int = 1,
        print_every: int = 100
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            x: Input tensor (n_samples, n_features)
            y: Target tensor (n_samples,)
            n_epochs: Maximum number of epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before stopping
            verbose: Verbosity level (0, 1, or 2)
            print_every: Print progress every N epochs
            
        Returns:
            Dictionary with training results and final model state
        """
        start_time = time.time()
        
        # Split data
        n_samples = x.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
        
        # Create dataloader
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        if verbose >= 1:
            print(f"Training on {n_train} samples, validating on {n_val} samples")
            print(f"Model: {type(self.model).__name__}")
            print("-" * 60)
        
        for epoch in range(n_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch, n_epochs)
            
            # Validate
            val_metrics = self.evaluate(x_val, y_val)
            
            # Check for improvement
            if val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose >= 1:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Print progress
            if verbose >= 1 and (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val MSE: {val_metrics['mse']:.6f} | "
                    f"Complexity: {train_metrics['complexity']:.2f} | "
                    f"λ: {train_metrics['complexity_weight']:.4f}"
                )
                if verbose >= 2:
                    print(f"  Current expr: {self.model.simplify()}")
        
        # Restore best model
        self.model.load_state_dict(best_state)
        
        # Final evaluation
        final_train_metrics = self.evaluate(x_train, y_train)
        final_val_metrics = self.evaluate(x_val, y_val)
        
        elapsed_time = time.time() - start_time
        
        if verbose >= 1:
            print("-" * 60)
            print(f"Training completed in {elapsed_time:.2f}s")
            print(f"Best epoch: {best_epoch + 1}")
            print(f"Final Train MSE: {final_train_metrics['mse']:.6f}")
            print(f"Final Val MSE: {final_val_metrics['mse']:.6f}")
            print(f"Final R²: {final_val_metrics['r2']:.4f}")
        
        return {
            "history": self.history,
            "best_epoch": best_epoch,
            "train_metrics": final_train_metrics,
            "val_metrics": final_val_metrics,
            "elapsed_time": elapsed_time,
            "n_epochs_trained": epoch + 1
        }


class LRScheduler:
    """
    Learning rate scheduler with multiple strategies.
    """
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str,
        **kwargs
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler
                - "cosine": Cosine annealing
                - "step": Step decay
                - "plateau": Reduce on plateau
                - "exponential": Exponential decay
            **kwargs: Additional arguments for scheduler
            
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 1000),
                eta_min=kwargs.get("eta_min", 1e-6)
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 200),
                gamma=kwargs.get("gamma", 0.5)
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 50),
                min_lr=kwargs.get("min_lr", 1e-6)
            )
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.999)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_symbolic_regressor(
    x: torch.Tensor,
    y: torch.Tensor,
    n_features: int,
    max_depth: int = 3,
    n_candidates: int = 5,
    n_epochs: int = 1000,
    learning_rate: float = 0.01,
    complexity_weight: float = 0.01,
    batch_size: int = 32,
    verbose: int = 1,
    **kwargs
) -> Tuple[SymbolicRegressor, Dict[str, Any]]:
    """
    Convenience function to train a symbolic regressor.
    
    Args:
        x: Input data (n_samples, n_features)
        y: Target data (n_samples,)
        n_features: Number of input features
        max_depth: Maximum expression tree depth
        n_candidates: Number of candidate expressions
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        complexity_weight: Complexity penalty weight
        batch_size: Batch size
        verbose: Verbosity level
        **kwargs: Additional arguments passed to trainer
        
    Returns:
        Tuple of (trained_model, training_results)
    """
    # Create model
    model = SymbolicRegressor(
        n_features=n_features,
        max_depth=max_depth,
        n_candidates=n_candidates,
        complexity_weight=complexity_weight
    )
    
    # Create trainer
    trainer = SymbolicRegressionTrainer(
        model=model,
        learning_rate=learning_rate,
        complexity_weight=complexity_weight,
        **kwargs
    )
    
    # Train
    results = trainer.fit(
        x, y,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return model, results

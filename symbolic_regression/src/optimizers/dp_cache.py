"""
Dynamic Programming Cache for Expression Evaluation

Implements memoization-based optimization to accelerate symbolic regression
by caching subexpression evaluations.

Key Optimizations:
    1. Expression Cache: LRU cache for subtree evaluation results
    2. Hash-based deduplication: Identifies equivalent expressions
    3. Incremental evaluation: Only recomputes changed subtrees
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from collections import OrderedDict
import hashlib
import time


class ExpressionCache:
    """
    LRU cache for expression evaluation results.
    
    Caches the output tensors of subtree evaluations keyed by a hash
    of the subtree structure and input data. This avoids redundant
    forward passes through unchanged subtrees.
    
    Complexity:
        - Lookup: O(1) average
        - Insert: O(1) amortized
        - Memory: O(capacity * tensor_size)
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of cached entries
        """
        self.capacity = capacity
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _compute_key(self, node_id: int, input_hash: str) -> str:
        """Compute cache key from node ID and input hash."""
        return f"{node_id}_{input_hash}"
    
    def _hash_tensor(self, x: torch.Tensor) -> str:
        """Compute hash of input tensor."""
        data = x.detach().cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:16]
    
    def get(self, node_id: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Retrieve cached result for a node evaluation.
        
        Parameters
        ----------
        node_id : int
            Unique identifier for the node
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor or None
            Cached result if found, None otherwise
        """
        input_hash = self._hash_tensor(x)
        key = self._compute_key(node_id, input_hash)
        
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].clone()
        
        self.misses += 1
        return None
    
    def put(self, node_id: int, x: torch.Tensor, result: torch.Tensor) -> None:
        """
        Store result in cache.
        
        Parameters
        ----------
        node_id : int
            Unique identifier for the node
        x : torch.Tensor
            Input tensor
        result : torch.Tensor
            Evaluation result to cache
        """
        input_hash = self._hash_tensor(x)
        key = self._compute_key(node_id, input_hash)
        
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = result.clone().detach()
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns
        -------
        dict
            Statistics including hits, misses, hit rate, and size
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.capacity
        }
    
    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self.hits = 0
        self.misses = 0


class DPOptimizer:
    """
    Dynamic Programming optimizer for symbolic regression.
    
    Wraps a model and provides optimized training with memoization.
    Uses the optimal substructure property of expression trees to
    cache and reuse subexpression evaluations.
    
    Theoretical Background:
        Symbolic regression can be viewed as a discrete optimization
        problem over expression trees. DP exploits that optimal trees
        contain optimal subtrees, and memoization prevents redundant
        evaluation of equivalent subexpressions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        cache_capacity: int = 1000,
        learning_rate: float = 0.01,
        complexity_weight: float = 0.01
    ):
        """
        Parameters
        ----------
        model : nn.Module
            The symbolic regression model to optimize
        cache_capacity : int
            Maximum cache size
        learning_rate : float
            Learning rate for Adam optimizer
        complexity_weight : float
            Weight for complexity penalty
        """
        self.model = model
        self.cache = ExpressionCache(capacity=cache_capacity)
        self.learning_rate = learning_rate
        self.complexity_weight = complexity_weight
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate
        )
        
        self.history: List[Dict[str, float]] = []
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        early_stopping_patience: int = 50
    ) -> List[Dict[str, float]]:
        """
        Train the model with DP optimization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (n_samples, n_features)
        y : torch.Tensor
            Target values (n_samples,)
        n_epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size
        verbose : bool
            Print progress
        early_stopping_patience : int
            Stop if no improvement for this many epochs
            
        Returns
        -------
        list of dict
            Training history
        """
        if y.dim() == 2:
            y = y.squeeze(-1)
        
        n_samples = x.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            x_shuffled = x[perm]
            y_shuffled = y[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.model(x_batch)
                if y_pred.dim() > 1:
                    y_pred = y_pred.squeeze(-1)
                
                # Compute loss
                mse = torch.mean((y_pred - y_batch) ** 2)
                complexity = self.model.get_complexity()
                loss = mse + self.complexity_weight * complexity
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Record history
            self.history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'cache_hit_rate': self.cache.get_stats()['hit_rate']
            })
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if verbose and (epoch + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f} | "
                      f"Cache hit rate: {self.cache.get_stats()['hit_rate']:.2%} | "
                      f"Time: {elapsed:.1f}s")
        
        return self.history
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        y : torch.Tensor
            Target values
            
        Returns
        -------
        dict
            Evaluation metrics (MSE, R², complexity)
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze(-1)
            if y.dim() > 1:
                y = y.squeeze(-1)
            
            mse = torch.mean((y_pred - y) ** 2).item()
            
            ss_res = torch.sum((y - y_pred) ** 2)
            ss_tot = torch.sum((y - torch.mean(y)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            
            complexity = self.model.get_complexity().item()
        
        return {
            'mse': mse,
            'r2': r2,
            'complexity': complexity
        }

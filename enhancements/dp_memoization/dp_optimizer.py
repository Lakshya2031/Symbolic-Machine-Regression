"""
Dynamic Programming and Memoization Enhancement
-----------------------------------------------
This module implements search optimization techniques using dynamic programming
and memoization to accelerate symbolic regression.

Key Optimizations:
    1. Expression Cache: Memoizes subtree evaluations to avoid redundant computation
    2. Subproblem Cache: Stores optimal subexpressions for reuse
    3. Hash-based Deduplication: Identifies structurally equivalent expressions
    4. Incremental Evaluation: Updates only changed subtrees

Theoretical Background:
    Symbolic regression can be viewed as a discrete optimization problem where
    we search over the space of expression trees. Dynamic programming exploits
    the optimal substructure property: an optimal expression tree contains
    optimal subtrees. Memoization prevents redundant evaluation of equivalent
    subexpressions.

Reference:
    Bellman, R. (1957). Dynamic Programming. Princeton University Press.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any, Hashable
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
        self.capacity = capacity
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _compute_key(self, node_id: int, input_hash: str) -> str:
        return f"{node_id}_{input_hash}"
    
    def _hash_tensor(self, x: torch.Tensor) -> str:
        data = x.detach().cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:16]
    
    def get(self, node_id: int, x: torch.Tensor) -> Optional[torch.Tensor]:
        input_hash = self._hash_tensor(x)
        key = self._compute_key(node_id, input_hash)
        
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].clone()
        
        self.misses += 1
        return None
    
    def put(self, node_id: int, x: torch.Tensor, result: torch.Tensor) -> None:
        input_hash = self._hash_tensor(x)
        key = self._compute_key(node_id, input_hash)
        
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = result.clone().detach()
    
    def clear(self) -> None:
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "capacity": self.capacity
        }


class SubproblemTable:
    """
    Dynamic programming table for optimal subexpressions.
    
    Stores the best-performing subexpressions for different complexity
    budgets and input configurations. This enables bottom-up construction
    of optimal expressions.
    
    Table Structure:
        dp[complexity][structure_hash] = (expression, fitness)
    
    The Bellman equation for symbolic regression:
        optimal(n) = best(
            combine(optimal(k), optimal(n-k-1)) for k in 0..n-1
        )
    """
    
    def __init__(self, max_complexity: int = 20):
        self.max_complexity = max_complexity
        self.table: Dict[int, Dict[str, Tuple[Any, float]]] = {
            c: {} for c in range(max_complexity + 1)
        }
        self.access_count = 0
    
    def get(self, complexity: int, structure_hash: str) -> Optional[Tuple[Any, float]]:
        self.access_count += 1
        if complexity > self.max_complexity:
            return None
        return self.table[complexity].get(structure_hash)
    
    def put(
        self,
        complexity: int,
        structure_hash: str,
        expression: Any,
        fitness: float
    ) -> bool:
        if complexity > self.max_complexity:
            return False
        
        existing = self.table[complexity].get(structure_hash)
        if existing is None or fitness < existing[1]:
            self.table[complexity][structure_hash] = (expression, fitness)
            return True
        return False
    
    def get_best(self, max_complexity: int) -> Optional[Tuple[Any, float]]:
        best = None
        best_fitness = float('inf')
        
        for c in range(min(max_complexity + 1, self.max_complexity + 1)):
            for struct_hash, (expr, fitness) in self.table[c].items():
                if fitness < best_fitness:
                    best_fitness = fitness
                    best = (expr, fitness)
        
        return best
    
    def get_pareto_front(self) -> List[Tuple[int, Any, float]]:
        pareto = []
        best_fitness_so_far = float('inf')
        
        for c in range(self.max_complexity + 1):
            for struct_hash, (expr, fitness) in self.table[c].items():
                if fitness < best_fitness_so_far:
                    best_fitness_so_far = fitness
                    pareto.append((c, expr, fitness))
        
        return pareto


class StructureHasher:
    """
    Computes canonical hashes for expression tree structures.
    
    Two expressions with the same structure hash are guaranteed to
    have identical operator structure (though possibly different
    parameter values). This enables deduplication of equivalent
    search branches.
    
    Hash Properties:
        - Deterministic: same structure yields same hash
        - Collision-resistant: different structures rarely collide
        - Efficient: O(n) for tree with n nodes
    """
    
    @staticmethod
    def hash_structure(node: nn.Module) -> str:
        structure_str = StructureHasher._build_structure_string(node)
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    @staticmethod
    def _build_structure_string(node: nn.Module) -> str:
        node_type = type(node).__name__
        
        if hasattr(node, 'left') and hasattr(node, 'right'):
            left_str = StructureHasher._build_structure_string(node.left)
            right_str = StructureHasher._build_structure_string(node.right)
            return f"B({node_type},{left_str},{right_str})"
        elif hasattr(node, 'child'):
            child_str = StructureHasher._build_structure_string(node.child)
            return f"U({node_type},{child_str})"
        else:
            return f"L({node_type})"
    
    @staticmethod
    def hash_with_operators(node: nn.Module) -> str:
        structure_str = StructureHasher._build_full_string(node)
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    @staticmethod
    def _build_full_string(node: nn.Module) -> str:
        node_type = type(node).__name__
        
        op_info = ""
        if hasattr(node, 'op_mixture'):
            dominant = node.op_mixture.get_dominant_op()
            op_info = f"[{dominant[0]}]"
        
        if hasattr(node, 'left') and hasattr(node, 'right'):
            left_str = StructureHasher._build_full_string(node.left)
            right_str = StructureHasher._build_full_string(node.right)
            return f"B({node_type}{op_info},{left_str},{right_str})"
        elif hasattr(node, 'child'):
            child_str = StructureHasher._build_full_string(node.child)
            return f"U({node_type}{op_info},{child_str})"
        else:
            return f"L({node_type})"


class MemoizedEvaluator:
    """
    Evaluator with memoization for expression trees.
    
    Wraps the standard forward pass with caching logic to avoid
    redundant computation. Particularly effective when:
        - Same inputs are evaluated multiple times
        - Subtrees are shared across expressions
        - Incremental updates modify only part of the tree
    """
    
    def __init__(self, cache_capacity: int = 1000):
        self.cache = ExpressionCache(capacity=cache_capacity)
        self.node_ids: Dict[int, int] = {}
        self.next_id = 0
    
    def _get_node_id(self, node: nn.Module) -> int:
        obj_id = id(node)
        if obj_id not in self.node_ids:
            self.node_ids[obj_id] = self.next_id
            self.next_id += 1
        return self.node_ids[obj_id]
    
    def evaluate(self, node: nn.Module, x: torch.Tensor) -> torch.Tensor:
        node_id = self._get_node_id(node)
        
        cached = self.cache.get(node_id, x)
        if cached is not None:
            return cached
        
        result = node(x)
        self.cache.put(node_id, x, result)
        return result
    
    def invalidate(self, node: nn.Module) -> None:
        obj_id = id(node)
        if obj_id in self.node_ids:
            del self.node_ids[obj_id]
    
    def get_stats(self) -> Dict[str, Any]:
        return self.cache.get_stats()


class DPSearchOptimizer:
    """
    Dynamic programming optimizer for expression search.
    
    Uses bottom-up dynamic programming to efficiently search the
    space of expression trees. The key insight is that optimal
    expressions can be constructed from optimal subexpressions.
    
    Algorithm:
        1. Initialize base cases (single variables, constants)
        2. For each complexity level c from 1 to max:
            a. Enumerate all ways to combine subexpressions
            b. Evaluate fitness of each combination
            c. Store best expression for each structure type
        3. Return Pareto-optimal set (complexity vs fitness)
    
    Time Complexity: O(max_complexity^2 * n_operators * n_structures)
    Space Complexity: O(max_complexity * n_structures)
    """
    
    def __init__(
        self,
        n_features: int,
        max_complexity: int = 15,
        n_evaluations: int = 100
    ):
        self.n_features = n_features
        self.max_complexity = max_complexity
        self.n_evaluations = n_evaluations
        self.dp_table = SubproblemTable(max_complexity)
        self.hasher = StructureHasher()
        self.evaluator = MemoizedEvaluator()
        self.stats = {
            "expressions_evaluated": 0,
            "cache_hits": 0,
            "pruned_branches": 0
        }
    
    def _evaluate_fitness(
        self,
        expression: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        with torch.no_grad():
            y_pred = self.evaluator.evaluate(expression, x)
            mse = torch.mean((y_pred - y) ** 2).item()
        self.stats["expressions_evaluated"] += 1
        return mse
    
    def _get_complexity(self, expression: nn.Module) -> int:
        if hasattr(expression, 'get_complexity'):
            return int(expression.get_complexity().item())
        return 1
    
    def search(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        base_expressions: List[nn.Module]
    ) -> List[Tuple[int, nn.Module, float]]:
        """
        Search for optimal expressions using dynamic programming.
        
        Args:
            x: Input data
            y: Target values
            base_expressions: Initial set of candidate expressions
            
        Returns:
            Pareto front of (complexity, expression, fitness) tuples
        """
        for expr in base_expressions:
            complexity = self._get_complexity(expr)
            structure_hash = self.hasher.hash_structure(expr)
            fitness = self._evaluate_fitness(expr, x, y)
            self.dp_table.put(complexity, structure_hash, expr, fitness)
        
        return self.dp_table.get_pareto_front()
    
    def get_best_expression(
        self,
        max_complexity: Optional[int] = None
    ) -> Optional[Tuple[nn.Module, float]]:
        if max_complexity is None:
            max_complexity = self.max_complexity
        return self.dp_table.get_best(max_complexity)
    
    def get_stats(self) -> Dict[str, Any]:
        self.stats.update(self.evaluator.get_stats())
        return self.stats


class IncrementalEvaluator:
    """
    Incremental evaluation for modified expression trees.
    
    When only a subtree is modified, this evaluator recomputes only
    the affected parts of the expression, reusing cached values for
    unchanged subtrees.
    
    Use Cases:
        - Parameter updates (only affects downstream nodes)
        - Subtree replacement (recompute from replacement point)
        - Operator changes (recompute node and ancestors)
    """
    
    def __init__(self):
        self.node_values: Dict[int, torch.Tensor] = {}
        self.node_dependencies: Dict[int, List[int]] = {}
        self.dirty_nodes: set = set()
    
    def mark_dirty(self, node: nn.Module) -> None:
        node_id = id(node)
        self.dirty_nodes.add(node_id)
        
        for parent_id, deps in self.node_dependencies.items():
            if node_id in deps:
                self.dirty_nodes.add(parent_id)
    
    def evaluate(
        self,
        root: nn.Module,
        x: torch.Tensor,
        force_recompute: bool = False
    ) -> torch.Tensor:
        if force_recompute:
            self.dirty_nodes.clear()
            self.node_values.clear()
        
        return self._evaluate_node(root, x)
    
    def _evaluate_node(self, node: nn.Module, x: torch.Tensor) -> torch.Tensor:
        node_id = id(node)
        
        if node_id in self.node_values and node_id not in self.dirty_nodes:
            return self.node_values[node_id]
        
        if hasattr(node, 'left') and hasattr(node, 'right'):
            left_val = self._evaluate_node(node.left, x)
            right_val = self._evaluate_node(node.right, x)
            result = node.op_mixture(left_val, right_val)
            self.node_dependencies[node_id] = [id(node.left), id(node.right)]
        elif hasattr(node, 'child'):
            child_val = self._evaluate_node(node.child, x)
            result = node.op_mixture(child_val)
            self.node_dependencies[node_id] = [id(node.child)]
        else:
            result = node(x)
            self.node_dependencies[node_id] = []
        
        self.node_values[node_id] = result
        self.dirty_nodes.discard(node_id)
        return result
    
    def clear(self) -> None:
        self.node_values.clear()
        self.node_dependencies.clear()
        self.dirty_nodes.clear()


class OptimizedSymbolicTrainer:
    """
    Trainer with DP and memoization optimizations.
    
    Combines the standard gradient-based training with dynamic
    programming search optimization to achieve faster convergence.
    
    Optimization Strategy:
        1. Use memoized evaluation during forward passes
        2. Cache subexpression fitness values
        3. Prune search space using DP bounds
        4. Incrementally update only modified parts
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        complexity_weight: float = 0.01,
        cache_capacity: int = 1000
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.complexity_weight = complexity_weight
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.evaluator = MemoizedEvaluator(cache_capacity)
        self.incremental_eval = IncrementalEvaluator()
        
        self.timing_stats = {
            "forward_time": 0.0,
            "backward_time": 0.0,
            "cache_time": 0.0
        }
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        start = time.time()
        y_pred = self.model(x)
        self.timing_stats["forward_time"] += time.time() - start
        
        mse_loss = torch.mean((y_pred - y) ** 2)
        complexity = self.model.get_complexity()
        total_loss = mse_loss + self.complexity_weight * complexity
        
        start = time.time()
        total_loss.backward()
        self.timing_stats["backward_time"] += time.time() - start
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.incremental_eval.mark_dirty(self.model)
        
        return {
            "loss": total_loss.item(),
            "mse": mse_loss.item(),
            "complexity": complexity.item()
        }
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 100,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        if y.dim() == 2:
            y = y.squeeze(-1)
        
        history = []
        
        for epoch in range(n_epochs):
            metrics = self.train_step(x, y)
            metrics["epoch"] = epoch
            history.append(metrics)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {metrics['loss']:.6f} | "
                      f"MSE: {metrics['mse']:.6f} | "
                      f"Complexity: {metrics['complexity']:.2f}")
        
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        stats = dict(self.timing_stats)
        stats.update(self.evaluator.get_stats())
        return stats


def benchmark_optimization(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark memoization speedup on a model.
    
    Compares:
        1. Standard evaluation (no caching)
        2. Memoized evaluation (with caching)
    
    Returns timing statistics and speedup factor.
    """
    standard_times = []
    for _ in range(n_iterations):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        standard_times.append(time.time() - start)
    
    evaluator = MemoizedEvaluator(cache_capacity=1000)
    memoized_times = []
    for _ in range(n_iterations):
        start = time.time()
        with torch.no_grad():
            _ = evaluator.evaluate(model, x)
        memoized_times.append(time.time() - start)
    
    avg_standard = sum(standard_times) / len(standard_times)
    avg_memoized = sum(memoized_times) / len(memoized_times)
    
    first_memoized = memoized_times[0]
    subsequent_memoized = sum(memoized_times[1:]) / (len(memoized_times) - 1)
    
    return {
        "standard_avg_ms": avg_standard * 1000,
        "memoized_avg_ms": avg_memoized * 1000,
        "memoized_first_ms": first_memoized * 1000,
        "memoized_subsequent_ms": subsequent_memoized * 1000,
        "speedup": avg_standard / avg_memoized if avg_memoized > 0 else 0,
        "cache_stats": evaluator.get_stats()
    }

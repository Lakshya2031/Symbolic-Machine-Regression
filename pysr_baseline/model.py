"""
Symbolic Regression Model
-------------------------
Main model class that combines symbolic tree nodes into a full computation
graph for learning f: R^n -> R.

The model maintains multiple candidate expression structures and uses
softmax-weighted combination for differentiable exploration of the
hypothesis space.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

from .nodes import (
    SymbolicNode,
    WeightedInputNode,
    UnaryOpNode,
    BinaryOpNode
)


class SymbolicExpression(nn.Module):
    """A single symbolic expression with fixed tree structure."""
    
    def __init__(
        self,
        n_features: int,
        max_depth: int = 3,
        structure_type: str = "binary_tree"
    ):
        super().__init__()
        self.n_features = n_features
        self.max_depth = max_depth
        self.structure_type = structure_type
        self.root = self._build_tree(max_depth, structure_type)
    
    def _build_tree(self, depth: int, structure_type: str) -> SymbolicNode:
        if structure_type == "binary_tree":
            return self._build_binary_tree(depth)
        elif structure_type == "unary_chain":
            return self._build_unary_chain(depth)
        elif structure_type == "mixed":
            return self._build_mixed_tree(depth)
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
    
    def _build_binary_tree(self, depth: int) -> SymbolicNode:
        if depth <= 1:
            return WeightedInputNode(self.n_features, mode="soft")
        left = self._build_binary_tree(depth - 1)
        right = self._build_binary_tree(depth - 1)
        return BinaryOpNode(left, right)
    
    def _build_unary_chain(self, depth: int) -> SymbolicNode:
        node = WeightedInputNode(self.n_features, mode="soft")
        for _ in range(depth):
            node = UnaryOpNode(node, include_identity=True)
        return node
    
    def _build_mixed_tree(self, depth: int) -> SymbolicNode:
        if depth <= 1:
            leaf = WeightedInputNode(self.n_features, mode="soft")
            return UnaryOpNode(leaf, include_identity=True)
        if depth % 3 == 0:
            child = self._build_mixed_tree(depth - 1)
            return UnaryOpNode(child, include_identity=True)
        else:
            left = self._build_mixed_tree(depth - 1)
            right = self._build_mixed_tree(depth - 1)
            return BinaryOpNode(left, right)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.root(x)
    
    def get_complexity(self) -> torch.Tensor:
        return self.root.get_complexity()
    
    def to_expression(self, var_names: Optional[List[str]] = None) -> str:
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        return self.root.to_expression(var_names)
    
    def simplify(self, var_names: Optional[List[str]] = None) -> str:
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        return self.root.simplify(var_names)


class SymbolicRegressor(nn.Module):
    """
    Symbolic regression model with multiple candidate expressions.
    
    Maintains a population of candidate expressions with different 
    structure types and complexity levels. Uses softmax-weighted
    combination for differentiable model selection.
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
        
        self.candidates = nn.ModuleList()
        for i in range(n_candidates):
            structure = structures[i % len(structures)]
            depth = max(1, max_depth - i // len(structures))
            expr = SymbolicExpression(n_features, depth, structure)
            self.candidates.append(expr)
        
        self.candidate_logits = nn.Parameter(torch.zeros(n_candidates))
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.candidate_logits, dim=0)
        outputs = torch.stack([cand(x) for cand in self.candidates], dim=1)
        result = torch.sum(outputs * weights, dim=1)
        result = result * self.output_scale + self.output_bias
        return result
    
    def get_complexity(self) -> torch.Tensor:
        weights = torch.softmax(self.candidate_logits, dim=0)
        complexities = torch.stack([cand.get_complexity() for cand in self.candidates])
        return torch.sum(weights * complexities)
    
    def get_candidate_weights(self) -> torch.Tensor:
        return torch.softmax(self.candidate_logits, dim=0)
    
    def get_dominant_candidate(self) -> Tuple[int, SymbolicExpression]:
        weights = self.get_candidate_weights()
        idx = torch.argmax(weights).item()
        return idx, self.candidates[idx]
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if complexity_weight is None:
            complexity_weight = self.complexity_weight
        
        y_pred = self(x)
        mse_loss = torch.mean((y_pred - y) ** 2)
        complexity = self.get_complexity()
        complexity_loss = complexity_weight * complexity
        total_loss = mse_loss + complexity_loss
        
        components = {
            "mse": mse_loss,
            "complexity": complexity,
            "complexity_loss": complexity_loss,
            "total": total_loss
        }
        return total_loss, components
    
    def to_expression(self, var_names: Optional[List[str]] = None) -> str:
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        
        weights = self.get_candidate_weights().detach().cpu().numpy()
        expressions = []
        
        for i, (w, cand) in enumerate(zip(weights, self.candidates)):
            if w > 0.01:
                expr = cand.to_expression(var_names)
                expressions.append(f"{w:.3f} * [{expr}]")
        
        scale = self.output_scale.item()
        bias = self.output_bias.item()
        
        result = " + ".join(expressions)
        if abs(scale - 1.0) > 0.01:
            result = f"{scale:.4f} * ({result})"
        if abs(bias) > 0.01:
            result = f"({result}) + {bias:.4f}"
        
        return result
    
    def simplify(self, var_names: Optional[List[str]] = None) -> str:
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        
        idx, dominant = self.get_dominant_candidate()
        expr = dominant.simplify(var_names)
        
        scale = self.output_scale.item()
        bias = self.output_bias.item()
        
        for nice_val in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            if abs(scale - nice_val) < 0.1:
                scale = nice_val
                break
        
        for nice_val in [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            if abs(bias - nice_val) < 0.1:
                bias = nice_val
                break
        
        if abs(scale - 1.0) > 0.01:
            expr = f"{scale:g} * {expr}"
        if abs(bias) > 0.01:
            expr = f"{expr} + {bias:g}"
        
        return expr

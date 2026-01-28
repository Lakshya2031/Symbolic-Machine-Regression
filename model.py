"""
Symbolic Regression Model Module
================================
Defines the main SymbolicRegressor model that combines symbolic tree nodes
into a full computation graph for learning f: R^n -> R.

Key features:
- Fixed maximum expression depth/complexity (PySR-style constraint)
- Multiple candidate expression structures
- Softmax-weighted operator selection for differentiable discrete choices
- Complexity-regularized loss function
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from nodes import (
    SymbolicNode, VariableNode, ConstantNode,
    UnaryOpNode, BinaryOpNode, PowerNode,
    LinearCombinationNode, WeightedInputNode
)


class SymbolicExpression(nn.Module):
    """
    A single symbolic expression with fixed structure.
    The structure is a tree of operator nodes with learnable parameters.
    """
    
    def __init__(
        self,
        n_features: int,
        max_depth: int = 3,
        structure_type: str = "binary_tree"
    ):
        """
        Args:
            n_features: Number of input features
            max_depth: Maximum depth of the expression tree
            structure_type: Type of tree structure to use
                - "binary_tree": Full binary tree
                - "unary_chain": Chain of unary operators
                - "mixed": Combination of binary and unary nodes
        """
        super().__init__()
        self.n_features = n_features
        self.max_depth = max_depth
        self.structure_type = structure_type
        
        # Build the expression tree
        self.root = self._build_tree(max_depth, structure_type)
    
    def _build_tree(self, depth: int, structure_type: str) -> SymbolicNode:
        """Build expression tree based on structure type."""
        if structure_type == "binary_tree":
            return self._build_binary_tree(depth)
        elif structure_type == "unary_chain":
            return self._build_unary_chain(depth)
        elif structure_type == "mixed":
            return self._build_mixed_tree(depth)
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
    
    def _build_binary_tree(self, depth: int) -> SymbolicNode:
        """Build a full binary tree of specified depth."""
        if depth <= 1:
            # Leaf: weighted input or constant
            return WeightedInputNode(self.n_features, mode="soft")
        
        # Internal node: binary operator with two children
        left = self._build_binary_tree(depth - 1)
        right = self._build_binary_tree(depth - 1)
        return BinaryOpNode(left, right)
    
    def _build_unary_chain(self, depth: int) -> SymbolicNode:
        """Build a chain of unary operators."""
        node = WeightedInputNode(self.n_features, mode="soft")
        for _ in range(depth):
            node = UnaryOpNode(node, include_identity=True)
        return node
    
    def _build_mixed_tree(self, depth: int) -> SymbolicNode:
        """Build a mixed tree with both binary and unary nodes."""
        if depth <= 1:
            # Leaf with optional unary transformation
            leaf = WeightedInputNode(self.n_features, mode="soft")
            return UnaryOpNode(leaf, include_identity=True)
        
        # 2/3 of nodes are binary, 1/3 are unary
        if depth % 3 == 0:
            # Unary node
            child = self._build_mixed_tree(depth - 1)
            return UnaryOpNode(child, include_identity=True)
        else:
            # Binary node
            left = self._build_mixed_tree(depth - 1)
            right = self._build_mixed_tree(depth - 1)
            return BinaryOpNode(left, right)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the expression tree."""
        return self.root(x)
    
    def get_complexity(self) -> torch.Tensor:
        """Get total complexity of the expression."""
        return self.root.get_complexity()
    
    def to_expression(self, var_names: Optional[List[str]] = None) -> str:
        """Convert to human-readable expression."""
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        return self.root.to_expression(var_names)
    
    def simplify(self, var_names: Optional[List[str]] = None) -> str:
        """Get simplified/discretized expression."""
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        return self.root.simplify(var_names)


class SymbolicRegressor(nn.Module):
    """
    Main symbolic regression model that learns f: R^n -> R.
    
    Architecture:
    - Multiple candidate symbolic expressions with different structures
    - Softmax-weighted combination of candidates
    - Complexity penalty for regularization
    
    This mirrors PySR's approach of maintaining a population of 
    candidate expressions with different complexity levels.
    """
    
    def __init__(
        self,
        n_features: int,
        max_depth: int = 3,
        n_candidates: int = 5,
        structures: Optional[List[str]] = None,
        complexity_weight: float = 0.01
    ):
        """
        Args:
            n_features: Number of input features
            max_depth: Maximum expression tree depth
            n_candidates: Number of candidate expressions
            structures: List of structure types for candidates
            complexity_weight: Weight λ for complexity penalty
        """
        super().__init__()
        self.n_features = n_features
        self.max_depth = max_depth
        self.n_candidates = n_candidates
        self.complexity_weight = complexity_weight
        
        # Default structure distribution
        if structures is None:
            structures = ["binary_tree", "mixed", "unary_chain"]
        
        # Build candidate expressions
        self.candidates = nn.ModuleList()
        for i in range(n_candidates):
            structure = structures[i % len(structures)]
            depth = max(1, max_depth - i // len(structures))
            expr = SymbolicExpression(n_features, depth, structure)
            self.candidates.append(expr)
        
        # Learnable weights for candidate selection
        self.candidate_logits = nn.Parameter(torch.zeros(n_candidates))
        
        # Optional output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: evaluate all candidates and return weighted sum.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output tensor of shape (batch_size,)
        """
        # Get candidate weights
        weights = torch.softmax(self.candidate_logits, dim=0)
        
        # Evaluate all candidates
        outputs = torch.stack([cand(x) for cand in self.candidates], dim=1)
        
        # Weighted combination
        result = torch.sum(outputs * weights, dim=1)
        
        # Apply output scaling
        result = result * self.output_scale + self.output_bias
        
        return result
    
    def get_complexity(self) -> torch.Tensor:
        """
        Get expected complexity based on candidate weights.
        
        Returns:
            Scalar complexity value
        """
        weights = torch.softmax(self.candidate_logits, dim=0)
        complexities = torch.stack([cand.get_complexity() for cand in self.candidates])
        return torch.sum(weights * complexities)
    
    def get_candidate_weights(self) -> torch.Tensor:
        """Return softmax weights for candidates."""
        return torch.softmax(self.candidate_logits, dim=0)
    
    def get_dominant_candidate(self) -> Tuple[int, SymbolicExpression]:
        """Return the dominant candidate (highest weight)."""
        weights = self.get_candidate_weights()
        idx = torch.argmax(weights).item()
        return idx, self.candidates[idx]
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss = MSE + λ * complexity.
        
        Args:
            x: Input tensor (batch_size, n_features)
            y: Target tensor (batch_size,)
            complexity_weight: Override default complexity weight
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if complexity_weight is None:
            complexity_weight = self.complexity_weight
        
        # Forward pass
        y_pred = self(x)
        
        # MSE loss
        mse_loss = torch.mean((y_pred - y) ** 2)
        
        # Complexity penalty
        complexity = self.get_complexity()
        complexity_loss = complexity_weight * complexity
        
        # Total loss
        total_loss = mse_loss + complexity_loss
        
        # Return components for logging
        components = {
            "mse": mse_loss,
            "complexity": complexity,
            "complexity_loss": complexity_loss,
            "total": total_loss
        }
        
        return total_loss, components
    
    def to_expression(self, var_names: Optional[List[str]] = None) -> str:
        """
        Convert model to human-readable expression.
        Shows weighted combination of all candidates.
        """
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
        """
        Get simplified expression using dominant candidate.
        Discretizes operator choices and prunes small weights.
        """
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        
        # Get dominant candidate
        idx, dominant = self.get_dominant_candidate()
        expr = dominant.simplify(var_names)
        
        # Apply output scaling if significant
        scale = self.output_scale.item()
        bias = self.output_bias.item()
        
        # Round scale to nice value
        for nice_val in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            if abs(scale - nice_val) < 0.1:
                scale = nice_val
                break
        
        # Round bias to nice value
        for nice_val in [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            if abs(bias - nice_val) < 0.1:
                bias = nice_val
                break
        
        if abs(scale - 1.0) > 0.01:
            expr = f"{scale:g} * {expr}"
        if abs(bias) > 0.01:
            expr = f"{expr} + {bias:g}"
        
        return expr


class MultiTermRegressor(nn.Module):
    """
    Alternative model structure using explicit additive terms.
    
    f(x) = c_0 + sum_i w_i * term_i(x)
    
    Each term is a symbolic expression. This is closer to how
    PySR builds expressions as sums of symbolic terms.
    """
    
    def __init__(
        self,
        n_features: int,
        n_terms: int = 5,
        max_depth: int = 2,
        complexity_weight: float = 0.01
    ):
        """
        Args:
            n_features: Number of input features
            n_terms: Number of additive terms
            max_depth: Maximum depth per term
            complexity_weight: Weight for complexity penalty
        """
        super().__init__()
        self.n_features = n_features
        self.n_terms = n_terms
        self.complexity_weight = complexity_weight
        
        # Build terms with varying structures
        self.terms = nn.ModuleList()
        structures = ["binary_tree", "mixed", "unary_chain"]
        for i in range(n_terms):
            structure = structures[i % len(structures)]
            term = SymbolicExpression(n_features, max_depth, structure)
            self.terms.append(term)
        
        # Learnable weights for each term
        self.term_weights = nn.Parameter(torch.zeros(n_terms))
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate as weighted sum of terms plus bias."""
        result = self.bias.expand(x.shape[0])
        for i, term in enumerate(self.terms):
            result = result + self.term_weights[i] * term(x)
        return result
    
    def get_complexity(self) -> torch.Tensor:
        """Get total complexity with sparsity penalty."""
        complexity = torch.tensor(0.0)
        
        # Term complexities weighted by term weights
        for i, term in enumerate(self.terms):
            w = torch.abs(self.term_weights[i])
            complexity = complexity + w * term.get_complexity()
        
        # L1 penalty on term weights (encourages sparsity)
        complexity = complexity + torch.sum(torch.abs(self.term_weights))
        
        return complexity
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss with complexity penalty."""
        if complexity_weight is None:
            complexity_weight = self.complexity_weight
        
        y_pred = self(x)
        mse_loss = torch.mean((y_pred - y) ** 2)
        complexity = self.get_complexity()
        complexity_loss = complexity_weight * complexity
        total_loss = mse_loss + complexity_loss
        
        return total_loss, {
            "mse": mse_loss,
            "complexity": complexity,
            "complexity_loss": complexity_loss,
            "total": total_loss
        }
    
    def simplify(self, var_names: Optional[List[str]] = None) -> str:
        """Get simplified expression with pruned terms."""
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n_features)]
        
        terms_str = []
        weights = self.term_weights.detach()
        
        for i, term in enumerate(self.terms):
            w = weights[i].item()
            if abs(w) > 0.1:  # Prune small weights
                expr = term.simplify(var_names)
                # Round weight
                w_rounded = round(w * 2) / 2
                if abs(w_rounded - 1.0) < 0.01:
                    terms_str.append(expr)
                elif abs(w_rounded + 1.0) < 0.01:
                    terms_str.append(f"-{expr}")
                else:
                    terms_str.append(f"{w_rounded:g}*{expr}")
        
        bias = self.bias.item()
        if abs(bias) > 0.1:
            bias_rounded = round(bias * 2) / 2
            terms_str.append(f"{bias_rounded:g}")
        
        return " + ".join(terms_str) if terms_str else "0"

"""
Symbolic Expression Module

Combines nodes into complete expression trees with different structures.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .nodes import (
    SymbolicNode,
    WeightedInputNode,
    UnaryOpNode,
    BinaryOpNode
)


class SymbolicExpression(nn.Module):
    """
    A single symbolic expression with fixed tree structure.
    
    The structure is determined at initialization and the operators/weights
    are learned during training.
    
    Structure Types:
        - "binary_tree": Full binary tree (most expressive)
        - "unary_chain": Chain of unary operators (for transcendental functions)
        - "mixed": Combination of binary and unary nodes
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
            structure_type: Type of tree structure ("binary_tree", "unary_chain", "mixed")
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
            return WeightedInputNode(self.n_features, mode="soft")
        
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

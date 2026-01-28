"""
Symbolic Tree Nodes Module
==========================
Defines the computation graph structure for symbolic expressions:
- Variable nodes: Input features x_i
- Constant nodes: Learnable constants (nn.Parameter)
- Operator nodes: Apply operators to child nodes

This module provides the building blocks for constructing 
explicit symbolic expressions with bounded complexity.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from operators import (
    BinaryOperatorMixture, 
    UnaryOperatorMixture, 
    LearnablePower,
    EPS
)


class SymbolicNode(nn.Module, ABC):
    """
    Abstract base class for all symbolic tree nodes.
    Each node represents a component of the symbolic expression.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate this node given input tensor x of shape (batch, n_features)."""
        pass
    
    @abstractmethod
    def get_complexity(self) -> torch.Tensor:
        """Return the complexity cost of this node."""
        pass
    
    @abstractmethod
    def to_expression(self, var_names: List[str]) -> str:
        """Convert this node to a human-readable expression string."""
        pass
    
    @abstractmethod
    def simplify(self, var_names: List[str]) -> str:
        """Return simplified/discretized expression after training."""
        pass


class VariableNode(SymbolicNode):
    """
    Variable node representing an input feature x_i.
    """
    
    def __init__(self, var_index: int):
        """
        Args:
            var_index: Index of the input variable (0-indexed)
        """
        super().__init__()
        self.var_index = var_index
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select the variable at var_index from input."""
        return x[:, self.var_index]
    
    def get_complexity(self) -> torch.Tensor:
        """Variables have minimal complexity."""
        return torch.tensor(0.5)
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return variable name."""
        if self.var_index < len(var_names):
            return var_names[self.var_index]
        return f"x{self.var_index}"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return variable name."""
        return self.to_expression(var_names)


class ConstantNode(SymbolicNode):
    """
    Constant node with a learnable value (nn.Parameter).
    """
    
    def __init__(self, init_value: float = 1.0):
        """
        Args:
            init_value: Initial value for the constant
        """
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the constant value expanded to batch size."""
        batch_size = x.shape[0]
        return self.value.expand(batch_size)
    
    def get_complexity(self) -> torch.Tensor:
        """Constants have minimal complexity."""
        return torch.tensor(1.0)
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return constant value as string."""
        return f"{self.value.item():.4f}"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return rounded constant value."""
        val = self.value.item()
        # Round to nice values if close
        for nice_val in [0, 1, -1, 2, -2, 0.5, -0.5, 3.14159, 2.71828]:
            if abs(val - nice_val) < 0.01:
                if nice_val == 3.14159:
                    return "π"
                elif nice_val == 2.71828:
                    return "e"
                else:
                    return f"{nice_val:g}"
        return f"{val:.4f}"


class UnaryOpNode(SymbolicNode):
    """
    Unary operator node that applies a softmax-weighted mixture
    of unary operators to its child node.
    """
    
    def __init__(self, child: SymbolicNode, include_identity: bool = True):
        """
        Args:
            child: Child node to apply operator to
            include_identity: Whether to include identity operator
        """
        super().__init__()
        self.child = child
        self.op_mixture = UnaryOperatorMixture(include_identity=include_identity)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply operator mixture to child's output."""
        child_output = self.child(x)
        return self.op_mixture(child_output)
    
    def get_complexity(self) -> torch.Tensor:
        """Return sum of operator complexity and child complexity."""
        return self.op_mixture.get_complexity() + self.child.get_complexity()
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return expression with operator weights."""
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        child_expr = self.child.to_expression(var_names)
        
        weights = self.op_mixture.get_weights().detach().cpu().numpy()
        weight_str = ", ".join([f"{self.op_mixture.ops[i][0]}:{w:.2f}" 
                               for i, w in enumerate(weights)])
        
        if op_symbol == "":
            return child_expr
        elif op_symbol == "-":
            return f"(-{child_expr})"
        elif op_symbol == "^2":
            return f"({child_expr})^2"
        else:
            return f"{op_symbol}({child_expr})"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return discretized expression with dominant operator."""
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        child_expr = self.child.simplify(var_names)
        
        if op_symbol == "":
            return child_expr
        elif op_symbol == "-":
            return f"(-{child_expr})"
        elif op_symbol == "^2":
            return f"({child_expr})^2"
        else:
            return f"{op_symbol}({child_expr})"


class BinaryOpNode(SymbolicNode):
    """
    Binary operator node that applies a softmax-weighted mixture
    of binary operators to two child nodes.
    """
    
    def __init__(self, left: SymbolicNode, right: SymbolicNode):
        """
        Args:
            left: Left child node
            right: Right child node
        """
        super().__init__()
        self.left = left
        self.right = right
        self.op_mixture = BinaryOperatorMixture()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply operator mixture to children's outputs."""
        left_output = self.left(x)
        right_output = self.right(x)
        return self.op_mixture(left_output, right_output)
    
    def get_complexity(self) -> torch.Tensor:
        """Return sum of operator complexity and children complexities."""
        return (self.op_mixture.get_complexity() + 
                self.left.get_complexity() + 
                self.right.get_complexity())
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return expression with operator weights."""
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        left_expr = self.left.to_expression(var_names)
        right_expr = self.right.to_expression(var_names)
        return f"({left_expr} {op_symbol} {right_expr})"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return discretized expression with dominant operator."""
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        left_expr = self.left.simplify(var_names)
        right_expr = self.right.simplify(var_names)
        return f"({left_expr} {op_symbol} {right_expr})"


class PowerNode(SymbolicNode):
    """
    Power node with learnable exponent: child^p
    where p is a trainable parameter.
    """
    
    def __init__(self, child: SymbolicNode, init_power: float = 2.0):
        """
        Args:
            child: Child node (base)
            init_power: Initial power value
        """
        super().__init__()
        self.child = child
        self.power_op = LearnablePower(init_power)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply power operation to child's output."""
        child_output = self.child(x)
        return self.power_op(child_output)
    
    def get_complexity(self) -> torch.Tensor:
        """Return sum of power complexity and child complexity."""
        return self.power_op.get_complexity() + self.child.get_complexity()
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return expression with power value."""
        child_expr = self.child.to_expression(var_names)
        power = self.power_op.get_power()
        return f"({child_expr})^{power:.3f}"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return simplified expression with rounded power."""
        child_expr = self.child.simplify(var_names)
        power = self.power_op.get_power()
        
        # Round to nice values if close
        for nice_val in [0, 0.5, 1, 2, 3, -1, -2, -0.5]:
            if abs(power - nice_val) < 0.1:
                if nice_val == 0:
                    return "1"
                elif nice_val == 1:
                    return child_expr
                elif nice_val == 2:
                    return f"({child_expr})^2"
                elif nice_val == 0.5:
                    return f"sqrt({child_expr})"
                elif nice_val == -1:
                    return f"(1/{child_expr})"
                elif nice_val == -0.5:
                    return f"(1/sqrt({child_expr}))"
                else:
                    return f"({child_expr})^{nice_val:g}"
        
        return f"({child_expr})^{power:.2f}"


class LinearCombinationNode(SymbolicNode):
    """
    Linear combination of multiple child nodes with learnable weights.
    Useful for combining multiple terms: w1*child1 + w2*child2 + ... + bias
    """
    
    def __init__(self, children: List[SymbolicNode], include_bias: bool = True):
        """
        Args:
            children: List of child nodes to combine
            include_bias: Whether to include a learnable bias term
        """
        super().__init__()
        self.children = nn.ModuleList(children)
        self.num_children = len(children)
        
        # Learnable weights for each child
        self.weights = nn.Parameter(torch.ones(self.num_children))
        
        # Optional bias term
        self.include_bias = include_bias
        if include_bias:
            self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of children outputs."""
        result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i, child in enumerate(self.children):
            result = result + self.weights[i] * child(x)
        if self.include_bias:
            result = result + self.bias
        return result
    
    def get_complexity(self) -> torch.Tensor:
        """Return sum of children complexities plus weight complexity."""
        complexity = torch.tensor(0.0)
        for child in self.children:
            complexity = complexity + child.get_complexity()
        # Add complexity for non-zero weights (L1 penalty style)
        complexity = complexity + torch.sum(torch.abs(self.weights)) * 0.5
        return complexity
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return expression showing linear combination."""
        terms = []
        for i, child in enumerate(self.children):
            w = self.weights[i].item()
            if abs(w) > 0.01:
                child_expr = child.to_expression(var_names)
                terms.append(f"{w:.4f}*{child_expr}")
        
        if self.include_bias:
            b = self.bias.item()
            if abs(b) > 0.01:
                terms.append(f"{b:.4f}")
        
        return " + ".join(terms) if terms else "0"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return simplified expression pruning small weights."""
        terms = []
        for i, child in enumerate(self.children):
            w = self.weights[i].item()
            if abs(w) > 0.1:  # Prune small weights
                child_expr = child.simplify(var_names)
                # Round weight to nice values
                w_rounded = round(w * 2) / 2  # Round to nearest 0.5
                if abs(w_rounded - 1.0) < 0.01:
                    terms.append(child_expr)
                elif abs(w_rounded + 1.0) < 0.01:
                    terms.append(f"-{child_expr}")
                else:
                    terms.append(f"{w_rounded:g}*{child_expr}")
        
        if self.include_bias:
            b = self.bias.item()
            if abs(b) > 0.1:
                b_rounded = round(b * 2) / 2
                terms.append(f"{b_rounded:g}")
        
        return " + ".join(terms) if terms else "0"


class WeightedInputNode(SymbolicNode):
    """
    Weighted selection/combination of input variables.
    Uses softmax to learn which input variable(s) are important.
    """
    
    def __init__(self, n_features: int, mode: str = "soft"):
        """
        Args:
            n_features: Number of input features
            mode: "soft" for softmax weighting, "linear" for linear combination
        """
        super().__init__()
        self.n_features = n_features
        self.mode = mode
        self.weights = nn.Parameter(torch.zeros(n_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return weighted combination of inputs.
        In soft mode: uses softmax weights (sums to 1)
        In linear mode: uses raw weights
        """
        if self.mode == "soft":
            w = torch.softmax(self.weights, dim=0)
        else:
            w = self.weights
        return torch.sum(x * w, dim=1)
    
    def get_complexity(self) -> torch.Tensor:
        """Return complexity based on weight entropy/sparsity."""
        if self.mode == "soft":
            w = torch.softmax(self.weights, dim=0)
            # Entropy of weight distribution (lower = simpler/more focused)
            entropy = -torch.sum(w * torch.log(w + EPS))
            return entropy + 0.5
        else:
            return torch.sum(torch.abs(self.weights)) * 0.5 + 0.5
    
    def get_dominant_input(self) -> int:
        """Return index of the dominant input variable."""
        if self.mode == "soft":
            w = torch.softmax(self.weights, dim=0)
        else:
            w = self.weights
        return torch.argmax(torch.abs(w)).item()
    
    def to_expression(self, var_names: List[str]) -> str:
        """Return expression showing weight distribution."""
        if self.mode == "soft":
            w = torch.softmax(self.weights, dim=0).detach().cpu().numpy()
        else:
            w = self.weights.detach().cpu().numpy()
        
        terms = []
        for i in range(self.n_features):
            if abs(w[i]) > 0.01:
                var_name = var_names[i] if i < len(var_names) else f"x{i}"
                terms.append(f"{w[i]:.3f}*{var_name}")
        return "(" + " + ".join(terms) + ")" if terms else "0"
    
    def simplify(self, var_names: List[str]) -> str:
        """Return simplified expression using dominant input."""
        idx = self.get_dominant_input()
        var_name = var_names[idx] if idx < len(var_names) else f"x{idx}"
        
        if self.mode == "soft":
            # In soft mode, just return dominant variable
            return var_name
        else:
            w = self.weights[idx].item()
            if abs(w - 1.0) < 0.1:
                return var_name
            elif abs(w + 1.0) < 0.1:
                return f"-{var_name}"
            else:
                return f"{w:.2f}*{var_name}"

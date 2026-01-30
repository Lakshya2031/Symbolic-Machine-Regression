"""
Symbolic Tree Nodes
-------------------
Building blocks for constructing symbolic expressions as computation graphs.
Each node type represents a component of the expression tree.

Node Types:
    - VariableNode: Input features x_i
    - ConstantNode: Learnable scalar constants
    - UnaryOpNode: Unary operator application
    - BinaryOpNode: Binary operator application
    - WeightedInputNode: Soft selection over input variables
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional

from .operators import (
    BinaryOperatorMixture,
    UnaryOperatorMixture,
    EPS
)


class SymbolicNode(nn.Module, ABC):
    """Abstract base class for symbolic tree nodes."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_complexity(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def to_expression(self, var_names: List[str]) -> str:
        pass
    
    @abstractmethod
    def simplify(self, var_names: List[str]) -> str:
        pass


class VariableNode(SymbolicNode):
    """Node representing an input variable."""
    
    def __init__(self, var_index: int):
        super().__init__()
        self.var_index = var_index
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.var_index]
    
    def get_complexity(self) -> torch.Tensor:
        return torch.tensor(0.5)
    
    def to_expression(self, var_names: List[str]) -> str:
        if self.var_index < len(var_names):
            return var_names[self.var_index]
        return f"x{self.var_index}"
    
    def simplify(self, var_names: List[str]) -> str:
        return self.to_expression(var_names)


class ConstantNode(SymbolicNode):
    """Node representing a learnable constant."""
    
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.value.expand(batch_size)
    
    def get_complexity(self) -> torch.Tensor:
        return torch.tensor(1.0)
    
    def to_expression(self, var_names: List[str]) -> str:
        return f"{self.value.item():.4f}"
    
    def simplify(self, var_names: List[str]) -> str:
        val = self.value.item()
        for nice_val in [0, 1, -1, 2, -2, 0.5, -0.5]:
            if abs(val - nice_val) < 0.01:
                return f"{nice_val:g}"
        return f"{val:.4f}"


class WeightedInputNode(SymbolicNode):
    """
    Node that computes a weighted combination of inputs.
    Supports soft selection (differentiable) or hard selection modes.
    """
    
    def __init__(self, n_features: int, mode: str = "soft"):
        super().__init__()
        self.n_features = n_features
        self.mode = mode
        self.feature_logits = nn.Parameter(torch.zeros(n_features))
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "soft":
            weights = torch.softmax(self.feature_logits, dim=0)
            result = torch.sum(x * weights, dim=1)
        else:
            idx = torch.argmax(self.feature_logits)
            result = x[:, idx]
        return result * self.scale
    
    def get_complexity(self) -> torch.Tensor:
        return torch.tensor(1.0)
    
    def get_dominant_feature(self) -> int:
        return torch.argmax(self.feature_logits).item()
    
    def to_expression(self, var_names: List[str]) -> str:
        weights = torch.softmax(self.feature_logits, dim=0).detach().cpu().numpy()
        scale = self.scale.item()
        terms = []
        for i, w in enumerate(weights):
            if w > 0.01:
                name = var_names[i] if i < len(var_names) else f"x{i}"
                terms.append(f"{scale * w:.3f}*{name}")
        return "(" + " + ".join(terms) + ")" if terms else "0"
    
    def simplify(self, var_names: List[str]) -> str:
        idx = self.get_dominant_feature()
        scale = self.scale.item()
        name = var_names[idx] if idx < len(var_names) else f"x{idx}"
        if abs(scale - 1.0) < 0.01:
            return name
        return f"{scale:.4f}*{name}"


class UnaryOpNode(SymbolicNode):
    """Node that applies a unary operator mixture to its child."""
    
    def __init__(self, child: SymbolicNode, include_identity: bool = True):
        super().__init__()
        self.child = child
        self.op_mixture = UnaryOperatorMixture(include_identity=include_identity)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        child_output = self.child(x)
        return self.op_mixture(child_output)
    
    def get_complexity(self) -> torch.Tensor:
        return self.op_mixture.get_complexity() + self.child.get_complexity()
    
    def to_expression(self, var_names: List[str]) -> str:
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        child_expr = self.child.to_expression(var_names)
        if op_symbol == "":
            return child_expr
        elif op_symbol == "-":
            return f"(-{child_expr})"
        elif op_symbol == "^2":
            return f"({child_expr})^2"
        else:
            return f"{op_symbol}({child_expr})"
    
    def simplify(self, var_names: List[str]) -> str:
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
    """Node that applies a binary operator mixture to two children."""
    
    def __init__(self, left: SymbolicNode, right: SymbolicNode):
        super().__init__()
        self.left = left
        self.right = right
        self.op_mixture = BinaryOperatorMixture()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_out = self.left(x)
        right_out = self.right(x)
        return self.op_mixture(left_out, right_out)
    
    def get_complexity(self) -> torch.Tensor:
        return (self.op_mixture.get_complexity() + 
                self.left.get_complexity() + 
                self.right.get_complexity())
    
    def to_expression(self, var_names: List[str]) -> str:
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        left_expr = self.left.to_expression(var_names)
        right_expr = self.right.to_expression(var_names)
        return f"({left_expr} {op_symbol} {right_expr})"
    
    def simplify(self, var_names: List[str]) -> str:
        op_name, op_symbol, _ = self.op_mixture.get_dominant_op()
        left_expr = self.left.simplify(var_names)
        right_expr = self.right.simplify(var_names)
        return f"({left_expr} {op_symbol} {right_expr})"


class PowerNode(SymbolicNode):
    """Node that applies x^p with learnable exponent p."""
    
    def __init__(self, child: SymbolicNode, init_power: float = 2.0):
        super().__init__()
        self.child = child
        self.power = nn.Parameter(torch.tensor(init_power))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        child_out = self.child(x)
        return torch.exp(self.power * torch.log(torch.abs(child_out) + EPS))
    
    def get_complexity(self) -> torch.Tensor:
        return torch.tensor(2.0) + self.child.get_complexity()
    
    def to_expression(self, var_names: List[str]) -> str:
        child_expr = self.child.to_expression(var_names)
        return f"({child_expr})^{self.power.item():.2f}"
    
    def simplify(self, var_names: List[str]) -> str:
        child_expr = self.child.simplify(var_names)
        p = self.power.item()
        for nice_p in [0, 1, 2, 3, -1, -2, 0.5]:
            if abs(p - nice_p) < 0.1:
                if nice_p == 0.5:
                    return f"sqrt({child_expr})"
                return f"({child_expr})^{int(nice_p)}"
        return f"({child_expr})^{p:.2f}"

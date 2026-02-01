"""
Operator Definitions - Pure PyTorch Implementation

Fixed operator space for symbolic regression including binary operators
(addition, subtraction, multiplication, protected division) and unary
operators (sin, cos, exp, protected log, protected sqrt).

All operators are differentiable PyTorch functions with numerical stability.
"""

import torch
import torch.nn as nn
from typing import Callable, List, Tuple

# Small constant for numerical stability
EPS = 1e-8

# Maximum output value to prevent overflow
MAX_OUTPUT = 1e6
MIN_OUTPUT = -1e6


def safe_clamp(x: torch.Tensor) -> torch.Tensor:
    """Clamp tensor to prevent overflow."""
    return torch.clamp(x, min=MIN_OUTPUT, max=MAX_OUTPUT)


# Binary Operators

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Addition: x + y"""
    return x + y


def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Subtraction: x - y"""
    return x - y


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiplication: x * y with clamping"""
    return safe_clamp(x * y)


def protected_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Protected division: x / y with numerical stability.
    Avoids division by zero by adding small epsilon.
    """
    y_safe = y + torch.sign(y) * EPS + EPS
    # Clamp y to prevent very small denominators
    y_safe = torch.where(torch.abs(y_safe) < 1e-4, torch.sign(y_safe) * 1e-4, y_safe)
    return safe_clamp(x / y_safe)


# Unary Operators

def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity function: x"""
    return x


def neg(x: torch.Tensor) -> torch.Tensor:
    """Negation: -x"""
    return -x


def sin_op(x: torch.Tensor) -> torch.Tensor:
    """Sine: sin(x)"""
    return torch.sin(x)


def cos_op(x: torch.Tensor) -> torch.Tensor:
    """Cosine: cos(x)"""
    return torch.cos(x)


def exp_op(x: torch.Tensor) -> torch.Tensor:
    """
    Protected exponential: exp(clamp(x))
    Clamps input to prevent overflow.
    """
    return torch.exp(torch.clamp(x, min=-5, max=5))  # Tighter clamp for stability


def protected_log(x: torch.Tensor) -> torch.Tensor:
    """
    Protected logarithm: log(|x| + eps)
    Uses absolute value to handle negative inputs.
    """
    return torch.log(torch.abs(x) + EPS)


def square(x: torch.Tensor) -> torch.Tensor:
    """Square: x^2"""
    return x ** 2


def sqrt_op(x: torch.Tensor) -> torch.Tensor:
    """
    Protected square root: sqrt(|x| + eps)
    Uses absolute value to handle negative inputs.
    """
    return torch.sqrt(torch.abs(x) + EPS)


def cube(x: torch.Tensor) -> torch.Tensor:
    """Cube: x^3 with clamping"""
    return safe_clamp(x ** 3)


def inv(x: torch.Tensor) -> torch.Tensor:
    """Protected inverse: 1 / (x + eps) with clamping"""
    x_safe = x + torch.sign(x) * EPS + EPS
    x_safe = torch.where(torch.abs(x_safe) < 1e-4, torch.sign(x_safe) * 1e-4, x_safe)
    return safe_clamp(1.0 / x_safe)


# Operator Registry

class OperatorRegistry:
    """
    Registry of available operators with complexity metadata.
    
    Each operator is defined as (name, function, symbol, complexity).
    Complexity values are used for regularization to prefer simpler expressions.
    """
    
    BINARY_OPS: List[Tuple[str, Callable, str, float]] = [
        ("add", add, "+", 1.0),
        ("sub", sub, "-", 1.0),
        ("mul", mul, "*", 1.5),
        ("div", protected_div, "/", 2.0),
    ]
    
    UNARY_OPS: List[Tuple[str, Callable, str, float]] = [
        ("identity", identity, "", 0.0),
        ("neg", neg, "-", 0.5),
        ("sin", sin_op, "sin", 3.0),
        ("cos", cos_op, "cos", 3.0),
        ("exp", exp_op, "exp", 4.0),
        ("log", protected_log, "log", 4.0),
        ("square", square, "^2", 2.0),
        ("sqrt", sqrt_op, "sqrt", 3.0),
    ]
    
    @classmethod
    def get_binary_ops(cls) -> List[Tuple[str, Callable, str, float]]:
        """Get all binary operators."""
        return cls.BINARY_OPS
    
    @classmethod
    def get_unary_ops(cls) -> List[Tuple[str, Callable, str, float]]:
        """Get all unary operators."""
        return cls.UNARY_OPS
    
    @classmethod
    def num_binary_ops(cls) -> int:
        """Number of binary operators."""
        return len(cls.BINARY_OPS)
    
    @classmethod
    def num_unary_ops(cls) -> int:
        """Number of unary operators."""
        return len(cls.UNARY_OPS)


# Differentiable Operator Mixtures

class BinaryOperatorMixture(nn.Module):
    """
    Differentiable mixture of binary operators.
    
    Uses softmax weights to compute a weighted combination of all binary
    operators' outputs. This enables gradient-based optimization over
    the discrete operator selection.
    
    This is the key innovation that allows gradient descent to work for
    symbolic regression - we relax the discrete operator choice into a
    continuous optimization problem.
    """
    
    def __init__(self):
        super().__init__()
        self.ops = OperatorRegistry.get_binary_ops()
        self.num_ops = len(self.ops)
        
        # Learnable logits for operator selection
        self.op_logits = nn.Parameter(torch.zeros(self.num_ops))
        
        # Complexity costs (non-learnable)
        self.register_buffer(
            "complexity_costs",
            torch.tensor([op[3] for op in self.ops])
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted mixture of all operators.
        
        Args:
            x: Left operand tensor
            y: Right operand tensor
            
        Returns:
            Weighted sum of operator outputs
        """
        weights = torch.softmax(self.op_logits, dim=0)
        result = torch.zeros_like(x)
        for i, (name, op_fn, symbol, _) in enumerate(self.ops):
            result = result + weights[i] * op_fn(x, y)
        return result
    
    def get_weights(self) -> torch.Tensor:
        """Get softmax weights for operators."""
        return torch.softmax(self.op_logits, dim=0)
    
    def get_dominant_op(self) -> Tuple[str, str, int]:
        """
        Get the operator with highest weight.
        
        Returns:
            Tuple of (name, symbol, index)
        """
        weights = self.get_weights()
        idx = torch.argmax(weights).item()
        return self.ops[idx][0], self.ops[idx][2], idx
    
    def get_complexity(self) -> torch.Tensor:
        """Get expected complexity based on weights."""
        weights = self.get_weights()
        return torch.sum(weights * self.complexity_costs)


class UnaryOperatorMixture(nn.Module):
    """
    Differentiable mixture of unary operators.
    
    Similar to BinaryOperatorMixture, uses softmax weights for
    continuous relaxation of discrete operator selection.
    """
    
    def __init__(self, include_identity: bool = True):
        super().__init__()
        self.ops = OperatorRegistry.get_unary_ops()
        
        if not include_identity:
            self.ops = [op for op in self.ops if op[0] != "identity"]
        
        self.num_ops = len(self.ops)
        self.op_logits = nn.Parameter(torch.zeros(self.num_ops))
        
        self.register_buffer(
            "complexity_costs",
            torch.tensor([op[3] for op in self.ops])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted mixture of all unary operators.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted sum of operator outputs
        """
        weights = torch.softmax(self.op_logits, dim=0)
        result = torch.zeros_like(x)
        for i, (name, op_fn, symbol, _) in enumerate(self.ops):
            result = result + weights[i] * op_fn(x)
        return result
    
    def get_weights(self) -> torch.Tensor:
        """Get softmax weights for operators."""
        return torch.softmax(self.op_logits, dim=0)
    
    def get_dominant_op(self) -> Tuple[str, str, int]:
        """
        Get the operator with highest weight.
        
        Returns:
            Tuple of (name, symbol, index)
        """
        weights = self.get_weights()
        idx = torch.argmax(weights).item()
        return self.ops[idx][0], self.ops[idx][2], idx
    
    def get_complexity(self) -> torch.Tensor:
        """Get expected complexity based on weights."""
        weights = self.get_weights()
        return torch.sum(weights * self.complexity_costs)

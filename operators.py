"""
Operator Definitions Module
===========================
Defines the fixed operator space for symbolic regression, including:
- Binary operators: +, -, *, / (protected)
- Unary operators: sin, cos, exp, log (protected)
- Power operator: x^p with learnable exponent

All operators are implemented as differentiable PyTorch functions.
"""

import torch
import torch.nn as nn
from typing import Callable, List, Tuple

# Numerical stability constant
EPS = 1e-8


# =============================================================================
# Binary Operators
# =============================================================================

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Addition operator."""
    return x + y


def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Subtraction operator."""
    return x - y


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiplication operator."""
    return x * y


def protected_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Protected division with numerical stability."""
    return x / (y + torch.sign(y) * EPS + EPS)


# =============================================================================
# Unary Operators
# =============================================================================

def sin_op(x: torch.Tensor) -> torch.Tensor:
    """Sine operator."""
    return torch.sin(x)


def cos_op(x: torch.Tensor) -> torch.Tensor:
    """Cosine operator."""
    return torch.cos(x)


def exp_op(x: torch.Tensor) -> torch.Tensor:
    """Exponential operator with clamping for stability."""
    return torch.exp(torch.clamp(x, min=-10, max=10))


def protected_log(x: torch.Tensor) -> torch.Tensor:
    """Protected logarithm: log(|x| + ε)."""
    return torch.log(torch.abs(x) + EPS)


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity operator (pass-through)."""
    return x


def neg(x: torch.Tensor) -> torch.Tensor:
    """Negation operator."""
    return -x


def square(x: torch.Tensor) -> torch.Tensor:
    """Square operator."""
    return x ** 2


def sqrt_op(x: torch.Tensor) -> torch.Tensor:
    """Protected square root: sqrt(|x| + ε)."""
    return torch.sqrt(torch.abs(x) + EPS)


# =============================================================================
# Power Operator (Learnable Exponent)
# =============================================================================

def power_op(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Power operator with learnable exponent: x^p
    Implemented as exp(p * log(|x| + ε)) for differentiability.
    Consistent with PySR's handling of power functions.
    """
    return torch.exp(p * torch.log(torch.abs(x) + EPS))


# =============================================================================
# Operator Registry
# =============================================================================

class OperatorRegistry:
    """
    Registry of all available operators with their metadata.
    This provides a fixed operator space for symbolic regression.
    """
    
    # Binary operators: (name, function, symbol, complexity)
    BINARY_OPS: List[Tuple[str, Callable, str, float]] = [
        ("add", add, "+", 1.0),
        ("sub", sub, "-", 1.0),
        ("mul", mul, "*", 1.0),
        ("div", protected_div, "/", 2.0),
    ]
    
    # Unary operators: (name, function, symbol, complexity)
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
        """Return list of binary operators."""
        return cls.BINARY_OPS
    
    @classmethod
    def get_unary_ops(cls) -> List[Tuple[str, Callable, str, float]]:
        """Return list of unary operators."""
        return cls.UNARY_OPS
    
    @classmethod
    def num_binary_ops(cls) -> int:
        """Return number of binary operators."""
        return len(cls.BINARY_OPS)
    
    @classmethod
    def num_unary_ops(cls) -> int:
        """Return number of unary operators."""
        return len(cls.UNARY_OPS)


# =============================================================================
# Softmax-Weighted Operator Mixture
# =============================================================================

class BinaryOperatorMixture(nn.Module):
    """
    Continuous relaxation of binary operator selection using softmax weights.
    Computes a weighted sum of all binary operators' outputs.
    """
    
    def __init__(self):
        super().__init__()
        self.ops = OperatorRegistry.get_binary_ops()
        self.num_ops = len(self.ops)
        # Learnable logits for operator selection
        self.op_logits = nn.Parameter(torch.zeros(self.num_ops))
        # Complexity costs for each operator
        self.register_buffer(
            "complexity_costs",
            torch.tensor([op[3] for op in self.ops])
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax-weighted mixture of binary operators.
        
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
        """Return softmax weights for operators."""
        return torch.softmax(self.op_logits, dim=0)
    
    def get_dominant_op(self) -> Tuple[str, str, int]:
        """Return the dominant operator (highest weight)."""
        weights = self.get_weights()
        idx = torch.argmax(weights).item()
        return self.ops[idx][0], self.ops[idx][2], idx
    
    def get_complexity(self) -> torch.Tensor:
        """Return expected complexity (weighted sum of operator complexities)."""
        weights = self.get_weights()
        return torch.sum(weights * self.complexity_costs)


class UnaryOperatorMixture(nn.Module):
    """
    Continuous relaxation of unary operator selection using softmax weights.
    Computes a weighted sum of all unary operators' outputs.
    """
    
    def __init__(self, include_identity: bool = True):
        super().__init__()
        self.ops = OperatorRegistry.get_unary_ops()
        if not include_identity:
            self.ops = [op for op in self.ops if op[0] != "identity"]
        self.num_ops = len(self.ops)
        # Learnable logits for operator selection
        self.op_logits = nn.Parameter(torch.zeros(self.num_ops))
        # Complexity costs for each operator
        self.register_buffer(
            "complexity_costs",
            torch.tensor([op[3] for op in self.ops])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax-weighted mixture of unary operators.
        
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
        """Return softmax weights for operators."""
        return torch.softmax(self.op_logits, dim=0)
    
    def get_dominant_op(self) -> Tuple[str, str, int]:
        """Return the dominant operator (highest weight)."""
        weights = self.get_weights()
        idx = torch.argmax(weights).item()
        return self.ops[idx][0], self.ops[idx][2], idx
    
    def get_complexity(self) -> torch.Tensor:
        """Return expected complexity (weighted sum of operator complexities)."""
        weights = self.get_weights()
        return torch.sum(weights * self.complexity_costs)


class LearnablePower(nn.Module):
    """
    Power operator with learnable exponent: x^p
    The exponent p is a trainable parameter.
    """
    
    def __init__(self, init_power: float = 1.0):
        super().__init__()
        self.power = nn.Parameter(torch.tensor(init_power))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply power operation with learnable exponent."""
        return power_op(x, self.power)
    
    def get_power(self) -> float:
        """Return current power value."""
        return self.power.item()
    
    def get_complexity(self) -> torch.Tensor:
        """Return complexity based on power deviation from simple values."""
        # Powers like 0, 1, 2, -1 are simpler
        simple_powers = torch.tensor([0.0, 1.0, 2.0, -1.0, 0.5])
        min_dist = torch.min(torch.abs(self.power - simple_powers))
        return 2.0 + min_dist  # Base complexity + deviation penalty

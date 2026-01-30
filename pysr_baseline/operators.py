"""
Operator Definitions
--------------------
Fixed operator space for symbolic regression including binary operators
(addition, subtraction, multiplication, protected division) and unary
operators (sin, cos, exp, protected log, protected sqrt).

All operators are implemented as differentiable PyTorch functions with
numerical stability guarantees.
"""

import torch
import torch.nn as nn
from typing import Callable, List, Tuple

EPS = 1e-8


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def protected_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / (y + torch.sign(y) * EPS + EPS)


def sin_op(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


def cos_op(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)


def exp_op(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, min=-10, max=10))


def protected_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.abs(x) + EPS)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def neg(x: torch.Tensor) -> torch.Tensor:
    return -x


def square(x: torch.Tensor) -> torch.Tensor:
    return x ** 2


def sqrt_op(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x) + EPS)


def power_op(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.exp(p * torch.log(torch.abs(x) + EPS))


class OperatorRegistry:
    """Registry of available operators with complexity metadata."""
    
    BINARY_OPS: List[Tuple[str, Callable, str, float]] = [
        ("add", add, "+", 1.0),
        ("sub", sub, "-", 1.0),
        ("mul", mul, "*", 1.0),
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
        return cls.BINARY_OPS
    
    @classmethod
    def get_unary_ops(cls) -> List[Tuple[str, Callable, str, float]]:
        return cls.UNARY_OPS
    
    @classmethod
    def num_binary_ops(cls) -> int:
        return len(cls.BINARY_OPS)
    
    @classmethod
    def num_unary_ops(cls) -> int:
        return len(cls.UNARY_OPS)


class BinaryOperatorMixture(nn.Module):
    """
    Continuous relaxation of binary operator selection.
    Uses softmax weights to compute a differentiable weighted combination
    of all binary operators' outputs.
    """
    
    def __init__(self):
        super().__init__()
        self.ops = OperatorRegistry.get_binary_ops()
        self.num_ops = len(self.ops)
        self.op_logits = nn.Parameter(torch.zeros(self.num_ops))
        self.register_buffer(
            "complexity_costs",
            torch.tensor([op[3] for op in self.ops])
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.op_logits, dim=0)
        result = torch.zeros_like(x)
        for i, (name, op_fn, symbol, _) in enumerate(self.ops):
            result = result + weights[i] * op_fn(x, y)
        return result
    
    def get_weights(self) -> torch.Tensor:
        return torch.softmax(self.op_logits, dim=0)
    
    def get_dominant_op(self) -> Tuple[str, str, int]:
        weights = self.get_weights()
        idx = torch.argmax(weights).item()
        return self.ops[idx][0], self.ops[idx][2], idx
    
    def get_complexity(self) -> torch.Tensor:
        weights = self.get_weights()
        return torch.sum(weights * self.complexity_costs)


class UnaryOperatorMixture(nn.Module):
    """
    Continuous relaxation of unary operator selection.
    Uses softmax weights to compute a differentiable weighted combination
    of all unary operators' outputs.
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
        weights = torch.softmax(self.op_logits, dim=0)
        result = torch.zeros_like(x)
        for i, (name, op_fn, symbol, _) in enumerate(self.ops):
            result = result + weights[i] * op_fn(x)
        return result
    
    def get_weights(self) -> torch.Tensor:
        return torch.softmax(self.op_logits, dim=0)
    
    def get_dominant_op(self) -> Tuple[str, str, int]:
        weights = self.get_weights()
        idx = torch.argmax(weights).item()
        return self.ops[idx][0], self.ops[idx][2], idx
    
    def get_complexity(self) -> torch.Tensor:
        weights = self.get_weights()
        return torch.sum(weights * self.complexity_costs)


class LearnablePower(nn.Module):
    """Power operator with learnable exponent."""
    
    def __init__(self, init_power: float = 1.0):
        super().__init__()
        self.power = nn.Parameter(torch.tensor(init_power))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return power_op(x, self.power)
    
    def get_power(self) -> float:
        return self.power.item()
    
    def get_complexity(self) -> torch.Tensor:
        simple_powers = torch.tensor([0.0, 1.0, 2.0, -1.0, 0.5])
        min_dist = torch.min(torch.abs(self.power - simple_powers))
        return 2.0 + min_dist

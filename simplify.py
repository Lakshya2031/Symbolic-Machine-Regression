"""
Expression Simplification Module
================================
Post-training tools for extracting and simplifying learned symbolic expressions:
- Discretize softmax-weighted operator selections
- Prune inactive branches
- Simplify constants to nice values
- Generate human-readable formulas

This mirrors PySR's simplify-optimize stage.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Any
import re
from model import SymbolicRegressor, MultiTermRegressor


class ExpressionSimplifier:
    """
    Tools for simplifying and extracting symbolic expressions from trained models.
    """
    
    # Common mathematical constants for pattern matching
    MATH_CONSTANTS = {
        3.14159: "π",
        2.71828: "e",
        1.41421: "√2",
        1.73205: "√3",
        0.69315: "ln(2)",
        1.0: "1",
        2.0: "2",
        0.5: "1/2",
        0.0: "0"
    }
    
    @staticmethod
    def discretize_operators(model: nn.Module, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Discretize all softmax-weighted operator selections.
        
        Args:
            model: Trained symbolic regression model
            threshold: Minimum weight to consider an operator "active"
            
        Returns:
            Dictionary mapping each operator mixture to its dominant operator
        """
        discretized = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'get_dominant_op'):
                op_name, op_symbol, idx = module.get_dominant_op()
                weights = module.get_weights().detach().cpu().numpy()
                max_weight = weights[idx]
                
                discretized[name] = {
                    "operator": op_name,
                    "symbol": op_symbol,
                    "weight": max_weight,
                    "confident": max_weight >= threshold,
                    "all_weights": {
                        module.ops[i][0]: w 
                        for i, w in enumerate(weights)
                    }
                }
        
        return discretized
    
    @staticmethod
    def round_constants(model: nn.Module, precision: float = 0.1) -> Dict[str, float]:
        """
        Round learned constants to nice values.
        
        Args:
            model: Trained model
            precision: Precision for rounding
            
        Returns:
            Dictionary of original and rounded constant values
        """
        constants = {}
        
        for name, param in model.named_parameters():
            if 'value' in name or 'bias' in name or 'weight' in name:
                original = param.item() if param.numel() == 1 else param.detach().cpu().numpy()
                
                if isinstance(original, float):
                    # Try to match to nice values
                    rounded = original
                    for nice_val, _ in ExpressionSimplifier.MATH_CONSTANTS.items():
                        if abs(original - nice_val) < precision:
                            rounded = nice_val
                            break
                    
                    # Otherwise round to precision
                    if rounded == original:
                        rounded = round(original / precision) * precision
                    
                    constants[name] = {
                        "original": original,
                        "rounded": rounded
                    }
        
        return constants
    
    @staticmethod
    def prune_inactive_branches(
        expr_str: str, 
        weight_threshold: float = 0.1
    ) -> str:
        """
        Simplify expression string by removing near-zero terms.
        
        Args:
            expr_str: Expression string
            weight_threshold: Threshold below which terms are pruned
            
        Returns:
            Simplified expression string
        """
        # Remove terms with very small coefficients
        # Pattern: coefficient * (expression)
        pattern = r'([+-]?\s*\d*\.?\d+)\s*\*\s*\([^)]+\)'
        
        def should_keep(match):
            coef = float(match.group(1).replace(' ', ''))
            return abs(coef) >= weight_threshold
        
        # Split by + and process terms
        terms = expr_str.split('+')
        kept_terms = []
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # Check if term has a small coefficient
            match = re.match(r'^([+-]?\s*\d*\.?\d+)\s*\*', term)
            if match:
                coef = float(match.group(1).replace(' ', ''))
                if abs(coef) >= weight_threshold:
                    kept_terms.append(term)
            else:
                kept_terms.append(term)
        
        return ' + '.join(kept_terms) if kept_terms else '0'
    
    @staticmethod
    def algebraic_simplify(expr_str: str) -> str:
        """
        Apply basic algebraic simplifications.
        
        Args:
            expr_str: Expression string
            
        Returns:
            Algebraically simplified expression
        """
        result = expr_str
        
        # Remove identity operations
        result = re.sub(r'\(\s*([^()]+)\s*\)\s*\^\s*1(?![0-9])', r'\1', result)  # x^1 -> x
        result = re.sub(r'1\s*\*\s*', '', result)  # 1 * x -> x
        result = re.sub(r'\s*\*\s*1(?![0-9])', '', result)  # x * 1 -> x
        result = re.sub(r'\s*\+\s*0(?![0-9.])', '', result)  # x + 0 -> x
        result = re.sub(r'0\s*\+\s*', '', result)  # 0 + x -> x
        
        # Simplify double negatives
        result = result.replace('--', '')
        result = result.replace('(-(-', '(')
        
        # Clean up extra parentheses
        result = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', result)
        
        # Remove leading +
        result = result.strip()
        if result.startswith('+'):
            result = result[1:].strip()
        
        return result
    
    @staticmethod
    def to_latex(expr_str: str) -> str:
        """
        Convert expression to LaTeX format.
        
        Args:
            expr_str: Expression string
            
        Returns:
            LaTeX formatted expression
        """
        latex = expr_str
        
        # Function names
        latex = latex.replace('sin(', r'\sin(')
        latex = latex.replace('cos(', r'\cos(')
        latex = latex.replace('exp(', r'\exp(')
        latex = latex.replace('log(', r'\log(')
        latex = latex.replace('sqrt(', r'\sqrt{')
        
        # Fix sqrt closing
        # This is a simple heuristic - may need improvement for nested cases
        latex = re.sub(r'\\sqrt\{([^}]+)\)', r'\\sqrt{\1}', latex)
        
        # Powers
        latex = re.sub(r'\^(\d+)', r'^{\1}', latex)
        latex = re.sub(r'\^(-?\d+\.?\d*)', r'^{\1}', latex)
        
        # Fractions for simple cases
        latex = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex)
        
        # Greek letters
        latex = latex.replace('π', r'\pi')
        
        # Multiplication
        latex = latex.replace('*', r' \cdot ')
        
        return latex
    
    @staticmethod
    def to_python_function(expr_str: str, var_names: List[str]) -> str:
        """
        Convert expression to executable Python function code.
        
        Args:
            expr_str: Expression string
            var_names: List of variable names
            
        Returns:
            Python function code as string
        """
        # Prepare expression for Python
        py_expr = expr_str
        
        # Replace math functions with numpy equivalents
        py_expr = py_expr.replace('sin(', 'np.sin(')
        py_expr = py_expr.replace('cos(', 'np.cos(')
        py_expr = py_expr.replace('exp(', 'np.exp(')
        py_expr = py_expr.replace('log(', 'np.log(np.abs(')
        
        # Fix log closing parenthesis
        py_expr = re.sub(r'np\.log\(np\.abs\(([^)]+)\)', r'np.log(np.abs(\1) + 1e-8)', py_expr)
        
        py_expr = py_expr.replace('sqrt(', 'np.sqrt(np.abs(')
        py_expr = re.sub(r'np\.sqrt\(np\.abs\(([^)]+)\)', r'np.sqrt(np.abs(\1) + 1e-8)', py_expr)
        
        # Powers
        py_expr = re.sub(r'\^(\d+\.?\d*)', r'**\1', py_expr)
        
        # Constants
        py_expr = py_expr.replace('π', 'np.pi')
        py_expr = py_expr.replace('e', 'np.e')
        
        # Generate function code
        var_args = ', '.join(var_names)
        
        code = f"""import numpy as np

def f({var_args}):
    \"\"\"
    Learned symbolic function.
    
    Args:
        {', '.join([f'{v}: input variable' for v in var_names])}
    
    Returns:
        Scalar output value
    \"\"\"
    return {py_expr}
"""
        return code


def extract_expression(
    model: nn.Module,
    var_names: Optional[List[str]] = None,
    simplify: bool = True,
    output_format: str = "string"
) -> Dict[str, Any]:
    """
    Extract and simplify expression from trained model.
    
    Args:
        model: Trained symbolic regression model
        var_names: Names for input variables
        simplify: Whether to apply simplification
        output_format: "string", "latex", or "python"
        
    Returns:
        Dictionary with expression in various formats
    """
    # Get variable names
    if var_names is None:
        n_features = model.n_features if hasattr(model, 'n_features') else 1
        var_names = [f"x{i}" for i in range(n_features)]
    
    # Get raw expression
    if simplify:
        expr_str = model.simplify(var_names)
    else:
        expr_str = model.to_expression(var_names)
    
    # Apply algebraic simplification
    simplified = ExpressionSimplifier.algebraic_simplify(expr_str)
    
    # Get operator decisions
    operator_info = ExpressionSimplifier.discretize_operators(model)
    
    # Get constant info
    constant_info = ExpressionSimplifier.round_constants(model)
    
    result = {
        "raw_expression": expr_str,
        "simplified": simplified,
        "operators": operator_info,
        "constants": constant_info
    }
    
    # Add requested format
    if output_format == "latex":
        result["latex"] = ExpressionSimplifier.to_latex(simplified)
    elif output_format == "python":
        result["python_code"] = ExpressionSimplifier.to_python_function(simplified, var_names)
    
    return result


def print_expression_report(
    model: nn.Module,
    var_names: Optional[List[str]] = None,
    show_details: bool = True
):
    """
    Print a comprehensive report of the learned expression.
    
    Args:
        model: Trained symbolic regression model
        var_names: Names for input variables
        show_details: Whether to show operator weights and constants
    """
    info = extract_expression(model, var_names, simplify=True)
    
    print("=" * 60)
    print("LEARNED SYMBOLIC EXPRESSION")
    print("=" * 60)
    print(f"\nSimplified Expression:")
    print(f"  f({', '.join(var_names or ['x'])}) = {info['simplified']}")
    
    print(f"\nLaTeX:")
    latex = ExpressionSimplifier.to_latex(info['simplified'])
    print(f"  ${latex}$")
    
    if show_details:
        print("\n" + "-" * 60)
        print("OPERATOR SELECTIONS")
        print("-" * 60)
        for name, op_info in info['operators'].items():
            conf_str = "✓" if op_info['confident'] else "?"
            print(f"\n  {name}:")
            print(f"    Selected: {op_info['operator']} ({op_info['symbol']}) "
                  f"[weight: {op_info['weight']:.3f}] {conf_str}")
            if not op_info['confident']:
                print("    All weights:")
                for op_name, weight in op_info['all_weights'].items():
                    if weight > 0.05:
                        print(f"      {op_name}: {weight:.3f}")
        
        print("\n" + "-" * 60)
        print("LEARNED CONSTANTS")
        print("-" * 60)
        for name, const_info in info['constants'].items():
            print(f"  {name}: {const_info['original']:.6f} -> {const_info['rounded']:.4f}")
    
    print("\n" + "=" * 60)


class ExpressionEvaluator:
    """
    Evaluate extracted expressions on new data.
    """
    
    def __init__(self, model: nn.Module, var_names: Optional[List[str]] = None):
        """
        Args:
            model: Trained symbolic regression model
            var_names: Names for input variables
        """
        self.model = model
        self.var_names = var_names
        
        # Get simplified expression
        info = extract_expression(model, var_names, simplify=True, output_format="python")
        self.expression = info['simplified']
        
        # Create executable function
        self._create_eval_function()
    
    def _create_eval_function(self):
        """Create evaluation function from expression."""
        import numpy as np
        
        # Build evaluation string
        eval_expr = self.expression
        eval_expr = eval_expr.replace('sin(', 'np.sin(')
        eval_expr = eval_expr.replace('cos(', 'np.cos(')
        eval_expr = eval_expr.replace('exp(', 'np.exp(')
        eval_expr = eval_expr.replace('log(', 'np.log(np.abs(')
        eval_expr = eval_expr.replace('sqrt(', 'np.sqrt(np.abs(')
        eval_expr = re.sub(r'\^(\d+\.?\d*)', r'**\1', eval_expr)
        eval_expr = eval_expr.replace('π', 'np.pi')
        
        self.eval_expr = eval_expr
    
    def evaluate(self, **kwargs) -> float:
        """
        Evaluate expression with given variable values.
        
        Args:
            **kwargs: Variable name -> value mapping
            
        Returns:
            Expression value
        """
        import numpy as np
        
        # Create local namespace with variables
        local_vars = {'np': np}
        local_vars.update(kwargs)
        
        return eval(self.eval_expr, {"__builtins__": {}}, local_vars)
    
    def evaluate_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate expression on batch of inputs using the model.
        
        Args:
            x: Input tensor (batch_size, n_features)
            
        Returns:
            Output tensor (batch_size,)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

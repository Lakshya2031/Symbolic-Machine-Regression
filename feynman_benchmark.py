"""
Feynman Benchmark for Symbolic Regression
==========================================
Tests the model on equations from the Feynman Lectures on Physics.
This is a standard benchmark for symbolic regression algorithms.

Reference: AI Feynman dataset (Udrescu & Tegmark, 2020)
"""

import torch
import numpy as np
import time
from typing import Callable, Tuple, List, Dict
from model import SymbolicRegressor, MultiTermRegressor
from trainer import SymbolicRegressionTrainer
from simplify import extract_expression

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# FEYNMAN EQUATIONS (subset of the benchmark)
# =============================================================================

FEYNMAN_EQUATIONS = {
    # Format: "name": (true_function, n_features, variable_names, data_ranges, true_formula)
    
    # I.6.2: Kinetic energy
    "I.6.2": (
        lambda x: 0.5 * x[:, 0] * x[:, 1]**2,  # E = 0.5 * m * v^2
        2,
        ["m", "v"],
        [(1, 5), (1, 5)],
        "0.5 * m * v^2"
    ),
    
    # I.8.14: Distance
    "I.8.14": (
        lambda x: torch.sqrt(x[:, 0]**2 + x[:, 1]**2),  # d = sqrt(x^2 + y^2)
        2,
        ["x", "y"],
        [(1, 5), (1, 5)],
        "sqrt(x^2 + y^2)"
    ),
    
    # I.9.18: Gravitational force ratio
    "I.9.18": (
        lambda x: x[:, 0] * x[:, 1] / (x[:, 2]**2),  # F ~ m1*m2/r^2
        3,
        ["m1", "m2", "r"],
        [(1, 5), (1, 5), (1, 5)],
        "m1 * m2 / r^2"
    ),
    
    # I.10.7: Simple addition with coefficient
    "I.10.7": (
        lambda x: x[:, 0] / (x[:, 1] * x[:, 2]),  # m0 / sqrt(1 - v^2/c^2) simplified
        3,
        ["m0", "v", "c"],
        [(1, 3), (0.1, 0.9), (1, 3)],
        "m0 / (v * c)"
    ),
    
    # I.11.19: Work done
    "I.11.19": (
        lambda x: x[:, 0] * x[:, 1] * torch.cos(x[:, 2]),  # W = F * d * cos(theta)
        3,
        ["F", "d", "theta"],
        [(1, 5), (1, 5), (0, 3.14)],
        "F * d * cos(theta)"
    ),
    
    # I.12.1: Electric field
    "I.12.1": (
        lambda x: x[:, 0] / (x[:, 1]**2),  # E = q / r^2
        2,
        ["q", "r"],
        [(1, 5), (1, 5)],
        "q / r^2"
    ),
    
    # I.12.11: Potential energy
    "I.12.11": (
        lambda x: x[:, 0] * x[:, 1] / x[:, 2],  # U = q1*q2/r
        3,
        ["q1", "q2", "r"],
        [(1, 5), (1, 5), (1, 5)],
        "q1 * q2 / r"
    ),
    
    # I.15.3x: Velocity addition (simplified)
    "I.15.3x": (
        lambda x: (x[:, 0] + x[:, 1]) / (1 + x[:, 0] * x[:, 1]),  # relativistic velocity
        2,
        ["u", "v"],
        [(0.1, 0.9), (0.1, 0.9)],
        "(u + v) / (1 + u*v)"
    ),
    
    # I.18.4: Simple harmonic motion period
    "I.18.4": (
        lambda x: 2 * 3.14159 * torch.sqrt(x[:, 0] / x[:, 1]),  # T = 2*pi*sqrt(m/k)
        2,
        ["m", "k"],
        [(1, 5), (1, 5)],
        "2*pi*sqrt(m/k)"
    ),
    
    # I.24.6: Power
    "I.24.6": (
        lambda x: 0.5 * x[:, 0] * x[:, 1]**2 * x[:, 2],  # P = 0.5*m*v^2*omega
        3,
        ["m", "v", "omega"],
        [(1, 3), (1, 3), (1, 3)],
        "0.5 * m * v^2 * omega"
    ),
    
    # I.27.6: Snell's law (ratio)
    "I.27.6": (
        lambda x: torch.sin(x[:, 0]) / torch.sin(x[:, 1]),  # n = sin(theta1)/sin(theta2)
        2,
        ["theta1", "theta2"],
        [(0.3, 1.2), (0.3, 1.2)],
        "sin(theta1) / sin(theta2)"
    ),
    
    # I.29.4: Wave number
    "I.29.4": (
        lambda x: x[:, 0] / x[:, 1],  # k = omega / c
        2,
        ["omega", "c"],
        [(1, 10), (1, 10)],
        "omega / c"
    ),
    
    # I.32.5: Power radiated (simplified)
    "I.32.5": (
        lambda x: x[:, 0]**2 * x[:, 1]**4,  # P ~ q^2 * a^4 / c^3 (simplified)
        2,
        ["q", "a"],
        [(1, 3), (1, 3)],
        "q^2 * a^4"
    ),
    
    # I.34.8: Angular momentum
    "I.34.8": (
        lambda x: x[:, 0] * x[:, 1] * x[:, 2],  # L = r * m * v
        3,
        ["r", "m", "v"],
        [(1, 5), (1, 5), (1, 5)],
        "r * m * v"
    ),
    
    # II.6.15a: Electric field from potential
    "II.6.15a": (
        lambda x: x[:, 0] / (4 * 3.14159 * x[:, 1]**2),  # E = q / (4*pi*r^2)
        2,
        ["q", "r"],
        [(1, 5), (1, 5)],
        "q / (4*pi*r^2)"
    ),
}


def generate_feynman_data(
    func: Callable,
    n_features: int,
    ranges: List[Tuple[float, float]],
    n_samples: int = 500,
    noise_ratio: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate data for a Feynman equation."""
    x = torch.zeros(n_samples, n_features)
    for i in range(n_features):
        low, high = ranges[i]
        x[:, i] = torch.rand(n_samples) * (high - low) + low
    
    y = func(x)
    
    if noise_ratio > 0:
        noise = torch.randn_like(y) * noise_ratio * y.std()
        y = y + noise
    
    return x, y


def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor
) -> Dict[str, float]:
    """Evaluate model performance."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        
        mse = torch.mean((y_pred - y)**2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(y_pred - y)).item()
        
        # R-squared
        ss_res = torch.sum((y - y_pred)**2)
        ss_tot = torch.sum((y - torch.mean(y))**2)
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        # Relative error
        rel_error = torch.mean(torch.abs(y_pred - y) / (torch.abs(y) + 1e-8)).item()
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rel_error": rel_error
    }


def run_feynman_benchmark(
    equations: List[str] = None,
    n_samples: int = 500,
    n_epochs: int = 2000,
    max_depth: int = 3,
    n_candidates: int = 5,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run benchmark on selected Feynman equations.
    
    Args:
        equations: List of equation names to test (None = all)
        n_samples: Number of training samples
        n_epochs: Training epochs
        max_depth: Expression tree depth
        n_candidates: Number of candidate expressions
        verbose: Print progress
        
    Returns:
        Dictionary of results for each equation
    """
    if equations is None:
        equations = list(FEYNMAN_EQUATIONS.keys())
    
    results = {}
    
    print("=" * 80)
    print("FEYNMAN BENCHMARK FOR SYMBOLIC REGRESSION")
    print("=" * 80)
    print(f"\nSettings: {n_samples} samples, {n_epochs} epochs, depth={max_depth}, candidates={n_candidates}")
    print(f"Testing {len(equations)} equations\n")
    
    for eq_name in equations:
        if eq_name not in FEYNMAN_EQUATIONS:
            print(f"Warning: Unknown equation {eq_name}, skipping")
            continue
        
        func, n_features, var_names, ranges, true_formula = FEYNMAN_EQUATIONS[eq_name]
        
        print("-" * 80)
        print(f"Equation {eq_name}: {true_formula}")
        print(f"Variables: {var_names}")
        print("-" * 80)
        
        # Generate data
        x_train, y_train = generate_feynman_data(func, n_features, ranges, n_samples)
        x_test, y_test = generate_feynman_data(func, n_features, ranges, n_samples // 5)
        
        # Create model
        model = SymbolicRegressor(
            n_features=n_features,
            max_depth=max_depth,
            n_candidates=n_candidates,
            complexity_weight=0.0005
        )
        
        # Train
        trainer = SymbolicRegressionTrainer(
            model=model,
            optimizer="adam",
            learning_rate=0.02,
            complexity_weight=0.0005,
            complexity_schedule="warmup"
        )
        
        start_time = time.time()
        train_results = trainer.fit(
            x_train, y_train,
            n_epochs=n_epochs,
            batch_size=64,
            verbose=0,
            early_stopping_patience=300
        )
        train_time = time.time() - start_time
        
        # Evaluate
        test_metrics = evaluate_model(model, x_test, y_test)
        
        # Get learned expression
        learned_expr = model.simplify(var_names)
        
        # Store results
        results[eq_name] = {
            "true_formula": true_formula,
            "learned_formula": learned_expr,
            "test_r2": test_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_rel_error": test_metrics["rel_error"],
            "train_time": train_time,
            "epochs_trained": train_results["n_epochs_trained"]
        }
        
        if verbose:
            print(f"  True formula:    {true_formula}")
            print(f"  Learned formula: {learned_expr}")
            print(f"  Test R²: {test_metrics['r2']:.6f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
            print(f"  Relative Error: {test_metrics['rel_error']*100:.2f}%")
            print(f"  Training time: {train_time:.1f}s ({train_results['n_epochs_trained']} epochs)")
            
            # Quality assessment
            if test_metrics['r2'] > 0.99:
                quality = "✓ EXCELLENT"
            elif test_metrics['r2'] > 0.95:
                quality = "○ GOOD"
            elif test_metrics['r2'] > 0.80:
                quality = "△ FAIR"
            else:
                quality = "✗ POOR"
            print(f"  Quality: {quality}")
        print()
    
    return results


def print_benchmark_summary(results: Dict[str, Dict]):
    """Print summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Equation':<12} {'R²':<10} {'RMSE':<12} {'Rel.Err%':<10} {'Time(s)':<10} {'Quality':<10}")
    print("-" * 80)
    
    r2_scores = []
    for eq_name, res in results.items():
        r2 = res['test_r2']
        r2_scores.append(r2)
        
        if r2 > 0.99:
            quality = "✓ EXCEL"
        elif r2 > 0.95:
            quality = "○ GOOD"
        elif r2 > 0.80:
            quality = "△ FAIR"
        else:
            quality = "✗ POOR"
        
        print(f"{eq_name:<12} {r2:<10.4f} {res['test_rmse']:<12.6f} {res['test_rel_error']*100:<10.2f} {res['train_time']:<10.1f} {quality:<10}")
    
    print("-" * 80)
    
    # Statistics
    r2_scores = np.array(r2_scores)
    print(f"\nStatistics across {len(results)} equations:")
    print(f"  Mean R²: {np.mean(r2_scores):.4f}")
    print(f"  Median R²: {np.median(r2_scores):.4f}")
    print(f"  Min R²: {np.min(r2_scores):.4f}")
    print(f"  Max R²: {np.max(r2_scores):.4f}")
    print(f"  Equations with R² > 0.99: {np.sum(r2_scores > 0.99)}/{len(r2_scores)}")
    print(f"  Equations with R² > 0.95: {np.sum(r2_scores > 0.95)}/{len(r2_scores)}")
    print(f"  Equations with R² > 0.80: {np.sum(r2_scores > 0.80)}/{len(r2_scores)}")


def quick_benchmark():
    """Run a quick benchmark on a subset of equations."""
    quick_equations = [
        "I.6.2",    # Kinetic energy: 0.5*m*v^2
        "I.8.14",   # Distance: sqrt(x^2 + y^2)
        "I.12.1",   # Electric field: q/r^2
        "I.29.4",   # Wave number: omega/c
        "I.34.8",   # Angular momentum: r*m*v
    ]
    
    print("\n" + "#" * 80)
    print("# QUICK BENCHMARK (5 equations)")
    print("#" * 80)
    
    results = run_feynman_benchmark(
        equations=quick_equations,
        n_samples=400,
        n_epochs=1500,
        max_depth=3,
        n_candidates=5,
        verbose=True
    )
    
    print_benchmark_summary(results)
    return results


def full_benchmark():
    """Run full benchmark on all equations."""
    print("\n" + "#" * 80)
    print("# FULL BENCHMARK (all equations)")
    print("#" * 80)
    
    results = run_feynman_benchmark(
        equations=None,  # All equations
        n_samples=500,
        n_epochs=2000,
        max_depth=3,
        n_candidates=6,
        verbose=True
    )
    
    print_benchmark_summary(results)
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        results = full_benchmark()
    else:
        results = quick_benchmark()
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("Run 'python feynman_benchmark.py full' for full benchmark")
    print("=" * 80)

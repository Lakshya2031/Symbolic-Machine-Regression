"""
Example: Symbolic Regression on Synthetic Data
==============================================
Demonstrates the PyTorch-based symbolic regression model on
synthetic scalar functions, printing both numerical error and
the recovered symbolic expression.

This example mirrors PySR's workflow:
1. Generate synthetic data from a known function
2. Train the symbolic regression model
3. Extract and simplify the learned expression
4. Compare with ground truth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

# Import our symbolic regression modules
from model import SymbolicRegressor, MultiTermRegressor
from trainer import SymbolicRegressionTrainer, train_symbolic_regressor
from simplify import extract_expression, print_expression_report


def generate_data(
    func: Callable,
    n_samples: int = 1000,
    n_features: int = 1,
    x_range: Tuple[float, float] = (-3, 3),
    noise_std: float = 0.0,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data from a known function.
    
    Args:
        func: Function to generate data from
        n_samples: Number of samples
        n_features: Number of input features
        x_range: Range for input values
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        Tuple of (x, y) tensors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random input
    x = torch.rand(n_samples, n_features) * (x_range[1] - x_range[0]) + x_range[0]
    
    # Compute output
    y = func(x)
    
    # Add noise
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    
    return x, y


def example_1_simple_polynomial():
    """
    Example 1: Learn f(x) = x^2 + 2x + 1
    A simple polynomial to verify basic functionality.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Polynomial")
    print("Ground truth: f(x) = x^2 + 2*x + 1")
    print("=" * 70)
    
    # Define ground truth function
    def true_func(x):
        return x[:, 0]**2 + 2*x[:, 0] + 1
    
    # Generate data
    x, y = generate_data(true_func, n_samples=500, n_features=1, x_range=(-5, 5))
    
    # Create model
    model = SymbolicRegressor(
        n_features=1,
        max_depth=3,
        n_candidates=3,
        complexity_weight=0.001
    )
    
    # Create trainer
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=0.05,
        complexity_weight=0.001,
        complexity_schedule="warmup"
    )
    
    # Train
    results = trainer.fit(
        x, y,
        n_epochs=2000,
        batch_size=64,
        verbose=1,
        print_every=500,
        early_stopping_patience=500
    )
    
    # Print results
    print_expression_report(model, var_names=["x"])
    
    # Final metrics
    final_metrics = trainer.evaluate(x, y)
    print(f"\nFinal MSE: {final_metrics['mse']:.6f}")
    print(f"Final R²: {final_metrics['r2']:.4f}")
    
    return model, results


def example_2_trigonometric():
    """
    Example 2: Learn f(x) = sin(x) + cos(2x)
    Tests trigonometric operator discovery.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Trigonometric Function")
    print("Ground truth: f(x) = sin(x) + cos(2*x)")
    print("=" * 70)
    
    def true_func(x):
        return torch.sin(x[:, 0]) + torch.cos(2*x[:, 0])
    
    x, y = generate_data(true_func, n_samples=800, n_features=1, x_range=(-3.14, 3.14))
    
    # Use MultiTermRegressor for additive structure
    model = MultiTermRegressor(
        n_features=1,
        n_terms=4,
        max_depth=2,
        complexity_weight=0.005
    )
    
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=0.02,
        complexity_weight=0.005,
        complexity_schedule="warmup"
    )
    
    results = trainer.fit(
        x, y,
        n_epochs=3000,
        batch_size=64,
        verbose=1,
        print_every=500,
        early_stopping_patience=500
    )
    
    print_expression_report(model, var_names=["x"])
    
    final_metrics = trainer.evaluate(x, y)
    print(f"\nFinal MSE: {final_metrics['mse']:.6f}")
    print(f"Final R²: {final_metrics['r2']:.4f}")
    
    return model, results


def example_3_multivariate():
    """
    Example 3: Learn f(x1, x2) = x1 * x2 + x1^2
    Tests multivariate function learning.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multivariate Function")
    print("Ground truth: f(x1, x2) = x1 * x2 + x1^2")
    print("=" * 70)
    
    def true_func(x):
        return x[:, 0] * x[:, 1] + x[:, 0]**2
    
    x, y = generate_data(true_func, n_samples=800, n_features=2, x_range=(-3, 3))
    
    model = SymbolicRegressor(
        n_features=2,
        max_depth=3,
        n_candidates=5,
        complexity_weight=0.001
    )
    
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=0.03,
        complexity_weight=0.001,
        complexity_schedule="warmup"
    )
    
    results = trainer.fit(
        x, y,
        n_epochs=2500,
        batch_size=64,
        verbose=1,
        print_every=500,
        early_stopping_patience=500
    )
    
    print_expression_report(model, var_names=["x1", "x2"])
    
    final_metrics = trainer.evaluate(x, y)
    print(f"\nFinal MSE: {final_metrics['mse']:.6f}")
    print(f"Final R²: {final_metrics['r2']:.4f}")
    
    return model, results


def example_4_physical_law():
    """
    Example 4: Learn Kepler's Third Law: T^2 = a^3
    (Reformulated as T = a^1.5)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Physical Law (Kepler's Third Law)")
    print("Ground truth: T = a^1.5 (orbital period vs semi-major axis)")
    print("=" * 70)
    
    def true_func(x):
        # T = a^1.5
        return torch.pow(torch.abs(x[:, 0]) + 1e-8, 1.5)
    
    x, y = generate_data(true_func, n_samples=500, n_features=1, x_range=(0.5, 5))
    
    model = SymbolicRegressor(
        n_features=1,
        max_depth=2,
        n_candidates=3,
        complexity_weight=0.01
    )
    
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=0.02,
        complexity_weight=0.01,
        complexity_schedule="linear"
    )
    
    results = trainer.fit(
        x, y,
        n_epochs=2000,
        batch_size=64,
        verbose=1,
        print_every=500,
        early_stopping_patience=500
    )
    
    print_expression_report(model, var_names=["a"])
    
    final_metrics = trainer.evaluate(x, y)
    print(f"\nFinal MSE: {final_metrics['mse']:.6f}")
    print(f"Final R²: {final_metrics['r2']:.4f}")
    
    return model, results


def example_5_noisy_data():
    """
    Example 5: Learn f(x) = 3*x + 2 with noisy data.
    Tests robustness to noise.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Noisy Data")
    print("Ground truth: f(x) = 3*x + 2 (with noise)")
    print("=" * 70)
    
    def true_func(x):
        return 3*x[:, 0] + 2
    
    x, y = generate_data(
        true_func, 
        n_samples=500, 
        n_features=1, 
        x_range=(-5, 5),
        noise_std=0.5
    )
    
    model = SymbolicRegressor(
        n_features=1,
        max_depth=2,
        n_candidates=3,
        complexity_weight=0.01
    )
    
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=0.05,
        complexity_weight=0.01
    )
    
    results = trainer.fit(
        x, y,
        n_epochs=1500,
        batch_size=64,
        verbose=1,
        print_every=500,
        early_stopping_patience=300
    )
    
    print_expression_report(model, var_names=["x"])
    
    final_metrics = trainer.evaluate(x, y)
    print(f"\nFinal MSE: {final_metrics['mse']:.6f}")
    print(f"Final R²: {final_metrics['r2']:.4f}")
    
    return model, results


def example_6_exponential():
    """
    Example 6: Learn f(x) = exp(-x^2)
    Gaussian-like function.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Exponential Function")
    print("Ground truth: f(x) = exp(-x^2)")
    print("=" * 70)
    
    def true_func(x):
        return torch.exp(-x[:, 0]**2)
    
    x, y = generate_data(true_func, n_samples=600, n_features=1, x_range=(-3, 3))
    
    model = SymbolicRegressor(
        n_features=1,
        max_depth=3,
        n_candidates=4,
        complexity_weight=0.002
    )
    
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=0.02,
        complexity_weight=0.002,
        complexity_schedule="warmup"
    )
    
    results = trainer.fit(
        x, y,
        n_epochs=3000,
        batch_size=64,
        verbose=1,
        print_every=500,
        early_stopping_patience=500
    )
    
    print_expression_report(model, var_names=["x"])
    
    final_metrics = trainer.evaluate(x, y)
    print(f"\nFinal MSE: {final_metrics['mse']:.6f}")
    print(f"Final R²: {final_metrics['r2']:.4f}")
    
    return model, results


def plot_results(
    model: torch.nn.Module,
    true_func: Callable,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 200,
    title: str = "Learned vs True Function"
):
    """
    Plot learned function vs ground truth.
    
    Args:
        model: Trained model
        true_func: Ground truth function
        x_range: Range for x axis
        n_points: Number of points to plot
        title: Plot title
    """
    # Generate test points
    x_test = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).numpy()
    
    y_true = true_func(x_test).numpy()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), y_true, 'b-', label='Ground Truth', linewidth=2)
    plt.plot(x_test.numpy(), y_pred, 'r--', label='Learned', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)
    plt.show()
    print("Plot saved as 'comparison_plot.png'")


def run_all_examples():
    """Run all examples."""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "SYMBOLIC REGRESSION EXAMPLES" + " " * 20 + "#")
    print("#" * 70)
    print("""
This demonstration shows the PyTorch-based symbolic regression model
learning various functions from data. The model uses:

- Softmax-weighted operator mixtures for differentiable discrete choices
- Gradient-based optimization (Adam) for all parameters
- Complexity penalties to encourage simpler expressions
- Post-training simplification to extract readable formulas

All updates are performed through backpropagation - no evolutionary
algorithms or non-differentiable optimization methods are used.
""")
    
    results = {}
    
    # Run each example
    results['polynomial'] = example_1_simple_polynomial()
    results['trigonometric'] = example_2_trigonometric()
    results['multivariate'] = example_3_multivariate()
    results['physical_law'] = example_4_physical_law()
    results['noisy'] = example_5_noisy_data()
    results['exponential'] = example_6_exponential()
    
    # Summary
    print("\n" + "#" * 70)
    print("#" + " " * 25 + "SUMMARY" + " " * 27 + "#")
    print("#" * 70)
    
    print("\n{:<20} {:<15} {:<15}".format("Example", "Final MSE", "Final R²"))
    print("-" * 50)
    
    for name, (model, res) in results.items():
        mse = res['val_metrics']['mse']
        r2 = res['val_metrics']['r2']
        print(f"{name:<20} {mse:<15.6f} {r2:<15.4f}")
    
    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run specific example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        if example_num == 1:
            example_1_simple_polynomial()
        elif example_num == 2:
            example_2_trigonometric()
        elif example_num == 3:
            example_3_multivariate()
        elif example_num == 4:
            example_4_physical_law()
        elif example_num == 5:
            example_5_noisy_data()
        elif example_num == 6:
            example_6_exponential()
        else:
            print(f"Unknown example number: {example_num}")
            print("Use 1-6 or no argument for all examples")
    else:
        run_all_examples()

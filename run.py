"""
Symbolic Regression Framework
=============================

Main entry point for running the symbolic regression system.

Example usage:
    python run.py              # Run demo with default settings
    python run.py --test       # Run test suite
    python run.py --benchmark  # Run performance benchmarks
"""

import argparse
import torch


def run_demo():
    """Demonstrate the symbolic regression system."""
    from pysr_baseline.model import SymbolicRegressor
    from pysr_baseline.trainer import SymbolicRegressionTrainer
    
    print("Symbolic Regression Demo")
    print("=" * 50)
    
    # Example 1: Quadratic function
    print("\nExample 1: Discovering y = x^2 + 2x + 1")
    print("-" * 50)
    
    torch.manual_seed(42)
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    y = x**2 + 2*x + 1
    
    model = SymbolicRegressor(n_features=1, max_depth=3, n_candidates=5)
    trainer = SymbolicRegressionTrainer(model, complexity_weight=0.01)
    
    trainer.fit(x, y, n_epochs=200, verbose=False)
    result = trainer.evaluate(x, y)
    
    print(f"Discovered expression: {model.simplify()}")
    print(f"MSE: {result['mse']:.6f}")
    print(f"R-squared: {result['r2']:.4f}")
    
    # Example 2: Trigonometric function
    print("\nExample 2: Discovering y = sin(x)")
    print("-" * 50)
    
    torch.manual_seed(123)
    x = torch.linspace(-3.14, 3.14, 100).unsqueeze(1)
    y = torch.sin(x)
    
    model = SymbolicRegressor(n_features=1, max_depth=2, n_candidates=3)
    trainer = SymbolicRegressionTrainer(model, complexity_weight=0.001)
    
    trainer.fit(x, y, n_epochs=200, verbose=False)
    result = trainer.evaluate(x, y)
    
    print(f"Discovered expression: {model.simplify()}")
    print(f"MSE: {result['mse']:.6f}")
    print(f"R-squared: {result['r2']:.4f}")
    
    # Example 3: Multivariate function
    print("\nExample 3: Discovering y = x1 * x2 + x1")
    print("-" * 50)
    
    torch.manual_seed(456)
    x = torch.randn(100, 2)
    y = x[:, 0:1] * x[:, 1:2] + x[:, 0:1]
    
    model = SymbolicRegressor(n_features=2, max_depth=3, n_candidates=5)
    trainer = SymbolicRegressionTrainer(model, complexity_weight=0.01)
    
    trainer.fit(x, y, n_epochs=200, verbose=False)
    result = trainer.evaluate(x, y)
    
    print(f"Discovered expression: {model.simplify()}")
    print(f"MSE: {result['mse']:.6f}")
    print(f"R-squared: {result['r2']:.4f}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully")


def run_tests():
    """Run the test suite."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "tests/test_suite.py"])


def run_benchmark():
    """Run performance benchmarks."""
    from benchmarks.benchmark_suite import run_benchmarks
    run_benchmarks()


def main():
    parser = argparse.ArgumentParser(
        description="Symbolic Regression Framework"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run the test suite"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run performance benchmarks"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.benchmark:
        run_benchmark()
    else:
        run_demo()


if __name__ == "__main__":
    main()

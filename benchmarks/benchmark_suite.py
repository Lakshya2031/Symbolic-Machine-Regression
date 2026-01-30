"""
Benchmark Suite for Symbolic Regression
---------------------------------------
Comprehensive benchmarking framework for comparing different symbolic
regression implementations and enhancements.

Benchmark Categories:
    1. Accuracy: MSE, R-squared on held-out test data
    2. Speed: Training time, inference time
    3. Complexity: Expression complexity of discovered formulas
    4. Scalability: Performance across different data sizes
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    dataset: str
    metrics: Dict[str, float]
    timing: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestFunction:
    """Defines a test function for benchmarking."""
    name: str
    func: Callable[[torch.Tensor], torch.Tensor]
    n_features: int
    formula: str
    complexity: int


class BenchmarkDatasets:
    """Collection of benchmark datasets for symbolic regression."""
    
    @staticmethod
    def get_polynomial_1d() -> TestFunction:
        def func(x):
            return x[:, 0]**2 + 2*x[:, 0] + 1
        return TestFunction(
            name="polynomial_1d",
            func=func,
            n_features=1,
            formula="x^2 + 2x + 1",
            complexity=5
        )
    
    @staticmethod
    def get_polynomial_2d() -> TestFunction:
        def func(x):
            return x[:, 0]**2 + x[:, 0]*x[:, 1] + x[:, 1]**2
        return TestFunction(
            name="polynomial_2d",
            func=func,
            n_features=2,
            formula="x0^2 + x0*x1 + x1^2",
            complexity=7
        )
    
    @staticmethod
    def get_trigonometric() -> TestFunction:
        def func(x):
            return torch.sin(x[:, 0]) + torch.cos(x[:, 0])
        return TestFunction(
            name="trigonometric",
            func=func,
            n_features=1,
            formula="sin(x) + cos(x)",
            complexity=4
        )
    
    @staticmethod
    def get_exponential() -> TestFunction:
        def func(x):
            return torch.exp(-x[:, 0]**2)
        return TestFunction(
            name="exponential",
            func=func,
            n_features=1,
            formula="exp(-x^2)",
            complexity=4
        )
    
    @staticmethod
    def get_rational() -> TestFunction:
        def func(x):
            return x[:, 0] / (1 + x[:, 0]**2)
        return TestFunction(
            name="rational",
            func=func,
            n_features=1,
            formula="x / (1 + x^2)",
            complexity=5
        )
    
    @staticmethod
    def get_kepler() -> TestFunction:
        def func(x):
            return x[:, 0] * x[:, 1]**2
        return TestFunction(
            name="kepler_third_law",
            func=func,
            n_features=2,
            formula="x0 * x1^2",
            complexity=4
        )
    
    @staticmethod
    def generate_data(
        test_func: TestFunction,
        n_samples: int = 200,
        noise_level: float = 0.0,
        x_range: Tuple[float, float] = (-3.0, 3.0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test data for a test function."""
        torch.manual_seed(42)
        
        n_train = int(0.8 * n_samples)
        n_test = n_samples - n_train
        
        x_train = torch.rand(n_train, test_func.n_features) * (x_range[1] - x_range[0]) + x_range[0]
        x_test = torch.rand(n_test, test_func.n_features) * (x_range[1] - x_range[0]) + x_range[0]
        
        y_train = test_func.func(x_train)
        y_test = test_func.func(x_test)
        
        if noise_level > 0:
            y_train = y_train + torch.randn_like(y_train) * noise_level
        
        return x_train, y_train, x_test, y_test
    
    @staticmethod
    def get_all_functions() -> List[TestFunction]:
        return [
            BenchmarkDatasets.get_polynomial_1d(),
            BenchmarkDatasets.get_polynomial_2d(),
            BenchmarkDatasets.get_trigonometric(),
            BenchmarkDatasets.get_exponential(),
            BenchmarkDatasets.get_rational(),
            BenchmarkDatasets.get_kepler()
        ]


class Benchmarker:
    """
    Main benchmarking class for comparing implementations.
    
    Usage:
        benchmarker = Benchmarker()
        benchmarker.add_implementation("baseline", baseline_factory, baseline_trainer)
        benchmarker.add_implementation("enhanced", enhanced_factory, enhanced_trainer)
        results = benchmarker.run_all()
        benchmarker.generate_report(results)
    """
    
    def __init__(self, n_runs: int = 3):
        self.n_runs = n_runs
        self.implementations: Dict[str, Dict[str, Any]] = {}
        self.results: List[BenchmarkResult] = []
    
    def add_implementation(
        self,
        name: str,
        model_factory: Callable[[int], nn.Module],
        trainer_factory: Callable[[nn.Module], Any]
    ) -> None:
        """Register an implementation for benchmarking."""
        self.implementations[name] = {
            "model_factory": model_factory,
            "trainer_factory": trainer_factory
        }
    
    def benchmark_single(
        self,
        impl_name: str,
        test_func: TestFunction,
        n_epochs: int = 100
    ) -> BenchmarkResult:
        """Run benchmark for a single implementation on a single dataset."""
        impl = self.implementations[impl_name]
        
        x_train, y_train, x_test, y_test = BenchmarkDatasets.generate_data(test_func)
        
        model = impl["model_factory"](test_func.n_features)
        trainer = impl["trainer_factory"](model)
        
        train_start = time.time()
        trainer.fit(x_train, y_train, n_epochs=n_epochs, verbose=False)
        train_time = time.time() - train_start
        
        inference_start = time.time()
        with torch.no_grad():
            y_pred = model(x_test)
        inference_time = time.time() - inference_start
        
        mse = torch.mean((y_pred - y_test) ** 2).item()
        
        y_mean = torch.mean(y_test)
        ss_tot = torch.sum((y_test - y_mean) ** 2)
        ss_res = torch.sum((y_test - y_pred) ** 2)
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        complexity = model.get_complexity().item() if hasattr(model, 'get_complexity') else 0.0
        
        try:
            expression = model.simplify() if hasattr(model, 'simplify') else "N/A"
        except:
            expression = "N/A"
        
        return BenchmarkResult(
            name=impl_name,
            dataset=test_func.name,
            metrics={
                "mse": mse,
                "r2": r2,
                "complexity": complexity
            },
            timing={
                "train_time": train_time,
                "inference_time": inference_time
            },
            metadata={
                "target_formula": test_func.formula,
                "discovered_formula": expression,
                "target_complexity": test_func.complexity,
                "n_epochs": n_epochs
            }
        )
    
    def run_all(self, n_epochs: int = 100) -> List[BenchmarkResult]:
        """Run all benchmarks for all implementations."""
        results = []
        test_functions = BenchmarkDatasets.get_all_functions()
        
        for impl_name in self.implementations:
            print(f"\nBenchmarking: {impl_name}")
            print("-" * 40)
            
            for test_func in test_functions:
                print(f"  Dataset: {test_func.name}...", end=" ")
                
                run_results = []
                for run in range(self.n_runs):
                    result = self.benchmark_single(impl_name, test_func, n_epochs)
                    run_results.append(result)
                
                avg_mse = sum(r.metrics["mse"] for r in run_results) / self.n_runs
                avg_time = sum(r.timing["train_time"] for r in run_results) / self.n_runs
                
                avg_result = BenchmarkResult(
                    name=impl_name,
                    dataset=test_func.name,
                    metrics={
                        "mse": avg_mse,
                        "mse_std": (sum((r.metrics["mse"] - avg_mse)**2 for r in run_results) / self.n_runs)**0.5,
                        "r2": sum(r.metrics["r2"] for r in run_results) / self.n_runs,
                        "complexity": sum(r.metrics["complexity"] for r in run_results) / self.n_runs
                    },
                    timing={
                        "train_time": avg_time,
                        "inference_time": sum(r.timing["inference_time"] for r in run_results) / self.n_runs
                    },
                    metadata=run_results[0].metadata
                )
                results.append(avg_result)
                print(f"MSE={avg_mse:.6f}, Time={avg_time:.2f}s")
        
        self.results = results
        return results
    
    def generate_report(self, results: Optional[List[BenchmarkResult]] = None) -> str:
        """Generate a formatted benchmark report."""
        if results is None:
            results = self.results
        
        lines = []
        lines.append("=" * 70)
        lines.append("BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Number of runs per test: {self.n_runs}")
        lines.append("")
        
        datasets = list(set(r.dataset for r in results))
        implementations = list(set(r.name for r in results))
        
        for dataset in datasets:
            lines.append("-" * 70)
            lines.append(f"Dataset: {dataset}")
            lines.append("-" * 70)
            
            dataset_results = [r for r in results if r.dataset == dataset]
            
            if dataset_results:
                lines.append(f"Target Formula: {dataset_results[0].metadata.get('target_formula', 'N/A')}")
                lines.append("")
            
            lines.append(f"{'Implementation':<20} {'MSE':<15} {'R2':<10} {'Time (s)':<12} {'Complexity':<12}")
            lines.append("-" * 70)
            
            for impl in implementations:
                impl_results = [r for r in dataset_results if r.name == impl]
                if impl_results:
                    r = impl_results[0]
                    lines.append(f"{impl:<20} {r.metrics['mse']:<15.6f} {r.metrics['r2']:<10.4f} "
                               f"{r.timing['train_time']:<12.2f} {r.metrics['complexity']:<12.2f}")
            
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        
        for impl in implementations:
            impl_results = [r for r in results if r.name == impl]
            avg_mse = sum(r.metrics["mse"] for r in impl_results) / len(impl_results)
            avg_time = sum(r.timing["train_time"] for r in impl_results) / len(impl_results)
            avg_r2 = sum(r.metrics["r2"] for r in impl_results) / len(impl_results)
            
            lines.append(f"{impl}:")
            lines.append(f"  Average MSE: {avg_mse:.6f}")
            lines.append(f"  Average R2: {avg_r2:.4f}")
            lines.append(f"  Average Training Time: {avg_time:.2f}s")
            lines.append("")
        
        report = "\n".join(lines)
        return report
    
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "n_runs": self.n_runs,
            "results": [
                {
                    "name": r.name,
                    "dataset": r.dataset,
                    "metrics": r.metrics,
                    "timing": r.timing,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def compare_implementations() -> None:
    """Run comparison between baseline and enhanced implementations."""
    from pysr_baseline.model import SymbolicRegressor
    from pysr_baseline.trainer import SymbolicRegressionTrainer
    
    def baseline_model_factory(n_features: int) -> nn.Module:
        return SymbolicRegressor(n_features=n_features, max_depth=3, n_candidates=5)
    
    def baseline_trainer_factory(model: nn.Module) -> Any:
        return SymbolicRegressionTrainer(model, complexity_weight=0.01)
    
    benchmarker = Benchmarker(n_runs=3)
    benchmarker.add_implementation("baseline", baseline_model_factory, baseline_trainer_factory)
    
    results = benchmarker.run_all(n_epochs=100)
    report = benchmarker.generate_report(results)
    print(report)
    
    benchmarker.save_results("benchmark_results.json")


if __name__ == "__main__":
    compare_implementations()

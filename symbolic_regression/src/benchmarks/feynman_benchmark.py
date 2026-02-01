"""
Feynman Benchmark Suite
=======================

Comprehensive benchmark suite for evaluating symbolic regression
on physics equations from the Feynman Lectures.

This reproduces results from the PySR paper:
    Cranmer, M. (2023). "Interpretable Machine Learning for Science 
    with PySR and SymbolicRegression.jl". arXiv:2305.01582

Key Metrics:
    - Accuracy (R², MSE)
    - Training time
    - Complexity of discovered expressions
    - Speedup from DP optimization
"""

import numpy as np
import torch
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data.dataset_utils import feynman_to_dataset, FEYNMAN_EQUATIONS


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    equation_id: str
    formula_true: str
    formula_discovered: str
    mse: float
    r2: float
    complexity: float
    train_time: float
    n_epochs: int
    success: bool  # R² > 0.95


class FeynmanBenchmark:
    """
    Feynman Benchmark Suite for Symbolic Regression.
    
    Evaluates model performance on physics equations with metrics
    comparable to PySR paper results.
    
    Example
    -------
    >>> from symbolic_regression import SymbolicRegressorModel, FeynmanBenchmark
    >>> 
    >>> # Run benchmark
    >>> benchmark = FeynmanBenchmark()
    >>> results = benchmark.run_all(model_class=SymbolicRegressorModel)
    >>> benchmark.print_summary(results)
    """
    
    # Core benchmark equations (from PySR paper)
    CORE_EQUATIONS = ['I.6.2', 'I.12.1', 'I.29.4']
    
    # Extended benchmark
    EXTENDED_EQUATIONS = ['I.6.2', 'I.6.2a', 'I.12.1', 'I.15.3x', 'I.29.4', 
                          'II.6.15a', 'II.11.3', 'trig.1']
    
    def __init__(
        self,
        n_samples: int = 500,
        n_epochs: int = 500,
        n_runs: int = 3,
        seed: int = 42
    ):
        """
        Parameters
        ----------
        n_samples : int
            Number of samples per equation
        n_epochs : int
            Training epochs per run
        n_runs : int
            Number of runs for averaging
        seed : int
            Base random seed
        """
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.n_runs = n_runs
        self.seed = seed
    
    def run_single(
        self,
        equation_id: str,
        model_class,
        model_kwargs: Optional[Dict] = None,
        verbose: bool = False
    ) -> BenchmarkResult:
        """
        Run benchmark on a single equation.
        
        Parameters
        ----------
        equation_id : str
            Feynman equation identifier
        model_class : class
            Model class (e.g., SymbolicRegressorModel)
        model_kwargs : dict, optional
            Additional arguments for model
        verbose : bool
            Print progress
            
        Returns
        -------
        BenchmarkResult
            Benchmark metrics
        """
        if not DEEPCHEM_AVAILABLE:
            raise ImportError("DeepChem required for benchmarks")
        
        # Get equation info
        eq_info = FEYNMAN_EQUATIONS[equation_id]
        
        # Create dataset
        dataset, info = feynman_to_dataset(
            equation_id, 
            n_samples=self.n_samples,
            seed=self.seed
        )
        
        # Default model kwargs
        if model_kwargs is None:
            model_kwargs = {}
        
        # Create model
        model = model_class(
            n_features=info['n_features'],
            max_depth=3,
            n_candidates=5,
            learning_rate=0.02,
            batch_size=64,
            **model_kwargs
        )
        
        # Train
        start_time = time.time()
        model.fit(dataset, nb_epoch=self.n_epochs)
        train_time = time.time() - start_time
        
        # Evaluate
        predictions = model.predict(dataset).squeeze()
        y_true = dataset.y.squeeze()
        
        mse = float(np.mean((predictions - y_true) ** 2))
        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        formula = model.get_formula(var_names=info['var_names'])
        complexity = model.get_complexity()
        
        if verbose:
            print(f"  {equation_id}: R²={r2:.4f}, Time={train_time:.1f}s")
            print(f"    True:       {info['formula']}")
            print(f"    Discovered: {formula}")
        
        return BenchmarkResult(
            equation_id=equation_id,
            formula_true=info['formula'],
            formula_discovered=formula,
            mse=mse,
            r2=r2,
            complexity=complexity,
            train_time=train_time,
            n_epochs=self.n_epochs,
            success=(r2 > 0.95)
        )
    
    def run_core(
        self,
        model_class,
        model_kwargs: Optional[Dict] = None,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run benchmark on core equations (I.6.2, I.12.1, I.29.4).
        
        These are the same equations used in the PySR paper for
        comparing symbolic regression methods.
        """
        results = []
        
        if verbose:
            print("="*60)
            print("CORE FEYNMAN BENCHMARK")
            print("="*60)
        
        for eq_id in self.CORE_EQUATIONS:
            result = self.run_single(eq_id, model_class, model_kwargs, verbose)
            results.append(result)
        
        return results
    
    def run_extended(
        self,
        model_class,
        model_kwargs: Optional[Dict] = None,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """Run benchmark on extended equation set."""
        results = []
        
        if verbose:
            print("="*60)
            print("EXTENDED FEYNMAN BENCHMARK")
            print("="*60)
        
        for eq_id in self.EXTENDED_EQUATIONS:
            result = self.run_single(eq_id, model_class, model_kwargs, verbose)
            results.append(result)
        
        return results
    
    def run_with_comparison(
        self,
        baseline_class,
        optimized_class,
        equations: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comparison between baseline and optimized models.
        
        Parameters
        ----------
        baseline_class : class
            Baseline model class
        optimized_class : class
            Optimized model class (e.g., with DP)
        equations : list of str, optional
            Equations to test. Default: core equations
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            {'baseline': results, 'optimized': results}
        """
        if equations is None:
            equations = self.CORE_EQUATIONS
        
        baseline_results = []
        optimized_results = []
        
        if verbose:
            print("="*60)
            print("BASELINE vs OPTIMIZED COMPARISON")
            print("="*60)
        
        for eq_id in equations:
            if verbose:
                print(f"\n{eq_id}:")
                print("-"*40)
            
            # Baseline
            if verbose:
                print("  Baseline:")
            baseline_result = self.run_single(
                eq_id, baseline_class, verbose=verbose
            )
            baseline_results.append(baseline_result)
            
            # Optimized
            if verbose:
                print("  Optimized:")
            optimized_result = self.run_single(
                eq_id, optimized_class, verbose=verbose
            )
            optimized_results.append(optimized_result)
            
            if verbose:
                speedup = baseline_result.train_time / optimized_result.train_time
                print(f"  Speedup: {speedup:.2f}x")
        
        return {
            'baseline': baseline_results,
            'optimized': optimized_results
        }
    
    @staticmethod
    def print_summary(results: List[BenchmarkResult]) -> None:
        """Print formatted summary of benchmark results."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Equation':<12} {'R²':>8} {'MSE':>10} {'Time':>8} {'Success':>8}")
        print("-"*70)
        
        for r in results:
            status = "✓" if r.success else "✗"
            print(f"{r.equation_id:<12} {r.r2:>8.4f} {r.mse:>10.6f} "
                  f"{r.train_time:>7.1f}s {status:>8}")
        
        print("-"*70)
        
        # Aggregates
        avg_r2 = np.mean([r.r2 for r in results])
        avg_time = np.mean([r.train_time for r in results])
        n_success = sum(1 for r in results if r.success)
        
        print(f"{'Average':<12} {avg_r2:>8.4f} {'-':>10} {avg_time:>7.1f}s "
              f"{n_success}/{len(results)}")
        print("="*70)
    
    @staticmethod
    def print_comparison(comparison: Dict[str, List[BenchmarkResult]]) -> None:
        """Print comparison between baseline and optimized."""
        baseline = comparison['baseline']
        optimized = comparison['optimized']
        
        print("\n" + "="*80)
        print("BASELINE vs OPTIMIZED COMPARISON")
        print("="*80)
        print(f"{'Equation':<12} {'Baseline R²':>12} {'Opt R²':>10} "
              f"{'Baseline Time':>14} {'Opt Time':>10} {'Speedup':>8}")
        print("-"*80)
        
        for b, o in zip(baseline, optimized):
            speedup = b.train_time / o.train_time
            print(f"{b.equation_id:<12} {b.r2:>12.4f} {o.r2:>10.4f} "
                  f"{b.train_time:>13.1f}s {o.train_time:>9.1f}s {speedup:>7.2f}x")
        
        print("-"*80)
        
        avg_baseline_time = np.mean([r.train_time for r in baseline])
        avg_opt_time = np.mean([r.train_time for r in optimized])
        avg_speedup = avg_baseline_time / avg_opt_time
        
        print(f"{'AVERAGE':<12} {'-':>12} {'-':>10} "
              f"{avg_baseline_time:>13.1f}s {avg_opt_time:>9.1f}s {avg_speedup:>7.2f}x")
        print("="*80)


def run_benchmark(
    model_class=None,
    equations: str = 'core',
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Convenience function to run benchmarks.
    
    Parameters
    ----------
    model_class : class, optional
        Model class. Default: SymbolicRegressorModel
    equations : str
        'core' or 'extended'
    verbose : bool
        Print progress
        
    Returns
    -------
    list of BenchmarkResult
    """
    if model_class is None:
        from models.symbolic_regressor import SymbolicRegressorModel
        model_class = SymbolicRegressorModel
    
    benchmark = FeynmanBenchmark()
    
    if equations == 'core':
        return benchmark.run_core(model_class, verbose=verbose)
    elif equations == 'extended':
        return benchmark.run_extended(model_class, verbose=verbose)
    else:
        raise ValueError(f"Unknown equation set: {equations}")


if __name__ == "__main__":
    print("Running Feynman Benchmark...")
    results = run_benchmark(equations='core')
    FeynmanBenchmark.print_summary(results)

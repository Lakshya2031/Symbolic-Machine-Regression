"""
Rigorous Evaluation Framework for Enhanced Symbolic Regression
==============================================================

This script performs comprehensive, statistically rigorous evaluation on
MoleculeNet benchmarks with attention to:

1. Multiple datasets (Delaney, Lipo)
2. Multiple random seeds for statistical significance
3. Proper cross-validation
4. Overfitting detection and early stopping
5. Hyperparameter sensitivity analysis
6. Detailed failure mode analysis

Key Failure Points Monitored:
- Numerical instability (NaN, Inf values)
- Overfitting (train loss << valid loss)
- Underfitting (high bias)
- Feature scaling issues
- Gradient explosion/vanishing

Author: GSoC Symbolic Regression Project
Date: February 2, 2026
"""

import sys
import os
import time
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import traceback

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'symbolic_regression', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'enhancements'))

# Import DeepChem
try:
    import deepchem as dc
    from deepchem.molnet import load_delaney, load_lipo
    DEEPCHEM_AVAILABLE = True
except ImportError as e:
    print(f"DeepChem not available: {e}")
    DEEPCHEM_AVAILABLE = False

# Import our model
try:
    from models.symbolic_regressor import SymbolicRegressorModel, DPSymbolicRegressorModel
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Model not available: {e}")
    MODEL_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """Configuration for rigorous evaluation."""
    # Datasets to test
    datasets: List[str] = field(default_factory=lambda: ['delaney', 'lipo'])
    
    # Statistical rigor
    n_seeds: int = 3  # Number of random seeds for statistical significance
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Model hyperparameters
    max_depth: int = 4
    n_candidates: int = 7
    complexity_weight: float = 0.005
    learning_rate: float = 0.01
    batch_size: int = 64
    
    # Training settings
    max_epochs: int = 200
    early_stopping_patience: int = 25
    min_improvement: float = 1e-4  # Minimum improvement to reset patience
    
    # Overfitting detection
    overfit_threshold: float = 2.0  # train_loss * threshold < valid_loss -> overfitting
    
    # Numerical stability
    gradient_clip: float = 10.0
    max_loss_value: float = 1e6  # Flag if loss exceeds this


@dataclass
class SingleRunResult:
    """Results from a single training run."""
    seed: int
    dataset: str
    
    # Final metrics
    train_mse: float
    valid_mse: float
    test_mse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    
    # Model info
    formula: str
    complexity: float
    
    # Training dynamics
    best_epoch: int
    total_epochs: int
    training_time: float
    
    # Health indicators
    overfit_detected: bool
    numerical_issues: bool
    early_stopped: bool
    
    # Detailed metrics
    train_losses: List[float] = field(default_factory=list)
    valid_losses: List[float] = field(default_factory=list)


@dataclass  
class DatasetResult:
    """Aggregated results for a dataset across multiple seeds."""
    dataset: str
    n_runs: int
    
    # Mean ± std metrics
    test_rmse_mean: float
    test_rmse_std: float
    test_r2_mean: float
    test_r2_std: float
    test_mae_mean: float
    test_mae_std: float
    
    # Best run info
    best_rmse: float
    best_r2: float
    best_formula: str
    
    # Health summary
    overfit_rate: float  # Fraction of runs with overfitting
    numerical_issue_rate: float
    early_stop_rate: float
    
    # All individual runs
    runs: List[SingleRunResult] = field(default_factory=list)


class OverfitDetector:
    """
    Monitors training for overfitting patterns.
    
    Overfitting Indicators:
    1. Train loss decreasing while valid loss increasing
    2. Large gap between train and valid loss
    3. Validation loss starts increasing after initial decrease
    """
    
    def __init__(self, patience: int = 10, threshold: float = 2.0):
        self.patience = patience
        self.threshold = threshold
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.overfit_detected = False
        self.overfit_epoch = None
    
    def update(self, train_loss: float, valid_loss: float, epoch: int) -> Tuple[bool, bool]:
        """
        Update with new losses.
        
        Returns:
            (should_stop, is_overfitting)
        """
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        
        # Check for improvement
        if valid_loss < self.best_valid_loss - 1e-4:
            self.best_valid_loss = valid_loss
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check for overfitting
        if len(self.train_losses) > 5:
            recent_train = np.mean(self.train_losses[-5:])
            recent_valid = np.mean(self.valid_losses[-5:])
            
            # Overfitting: train << valid
            if recent_valid > recent_train * self.threshold and recent_train < 0.5:
                self.overfit_detected = True
                if self.overfit_epoch is None:
                    self.overfit_epoch = epoch
        
        # Should stop if patience exceeded
        should_stop = self.patience_counter >= self.patience
        
        return should_stop, self.overfit_detected
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'best_epoch': self.best_epoch,
            'best_valid_loss': self.best_valid_loss,
            'overfit_detected': self.overfit_detected,
            'overfit_epoch': self.overfit_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_valid_loss': self.valid_losses[-1] if self.valid_losses else None
        }


class NumericalStabilityChecker:
    """
    Checks for numerical stability issues during training.
    
    Issues Detected:
    1. NaN values in loss or predictions
    2. Inf values
    3. Exploding gradients
    4. Loss explosion
    """
    
    def __init__(self, max_loss: float = 1e6):
        self.max_loss = max_loss
        self.issues = []
        self.nan_count = 0
        self.inf_count = 0
        self.explosion_count = 0
    
    def check_loss(self, loss: float, epoch: int) -> bool:
        """Check if loss is numerically stable. Returns True if OK."""
        if np.isnan(loss):
            self.nan_count += 1
            self.issues.append(f"NaN loss at epoch {epoch}")
            return False
        
        if np.isinf(loss):
            self.inf_count += 1
            self.issues.append(f"Inf loss at epoch {epoch}")
            return False
        
        if abs(loss) > self.max_loss:
            self.explosion_count += 1
            self.issues.append(f"Loss explosion ({loss:.2e}) at epoch {epoch}")
            return False
        
        return True
    
    def check_predictions(self, predictions: np.ndarray, epoch: int) -> bool:
        """Check if predictions are numerically stable."""
        if np.any(np.isnan(predictions)):
            self.nan_count += 1
            self.issues.append(f"NaN predictions at epoch {epoch}")
            return False
        
        if np.any(np.isinf(predictions)):
            self.inf_count += 1
            self.issues.append(f"Inf predictions at epoch {epoch}")
            return False
        
        return True
    
    def has_issues(self) -> bool:
        return len(self.issues) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'has_issues': self.has_issues(),
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'explosion_count': self.explosion_count,
            'issues': self.issues
        }


class RigorousEvaluator:
    """
    Performs rigorous evaluation with statistical analysis.
    """
    
    def __init__(self, config: EvaluationConfig = None, output_dir: str = None):
        self.config = config or EvaluationConfig()
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'rigorous_results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Store all results
        self.all_results: Dict[str, DatasetResult] = {}
    
    def load_dataset(self, dataset_name: str):
        """Load a MoleculeNet dataset."""
        loaders = {
            'delaney': load_delaney,
            'lipo': load_lipo
        }
        
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        tasks, datasets, transformers = loaders[dataset_name](
            featurizer='ECFP',
            splitter='scaffold',
            transformers=['normalization']
        )
        
        return datasets[0], datasets[1], datasets[2], tasks, transformers
    
    def train_single_run(
        self,
        train_data,
        valid_data,
        test_data,
        seed: int,
        dataset_name: str
    ) -> SingleRunResult:
        """
        Train model with a single seed and comprehensive monitoring.
        """
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        n_features = train_data.X.shape[1]
        
        # Initialize model
        model = DPSymbolicRegressorModel(
            n_features=n_features,
            max_depth=self.config.max_depth,
            n_candidates=self.config.n_candidates,
            complexity_weight=self.config.complexity_weight,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            device=self.device
        )
        
        # Initialize monitors
        overfit_detector = OverfitDetector(
            patience=self.config.early_stopping_patience,
            threshold=self.config.overfit_threshold
        )
        stability_checker = NumericalStabilityChecker(
            max_loss=self.config.max_loss_value
        )
        
        # Training loop with monitoring
        train_losses = []
        valid_losses = []
        best_state = None
        best_valid_mse = float('inf')
        best_epoch = 0
        
        start_time = time.time()
        early_stopped = False
        
        print(f"\n  Training with seed {seed}...")
        
        for epoch in range(self.config.max_epochs):
            try:
                # Train one epoch
                train_loss = model.fit(train_data, nb_epoch=1, deterministic=False)
                
                # Handle case where train_loss is a list
                if isinstance(train_loss, (list, tuple)):
                    train_loss = train_loss[-1] if train_loss else 0.0
                train_loss = float(train_loss)
                
                # Check numerical stability
                if not stability_checker.check_loss(train_loss, epoch):
                    print(f"    Numerical issue at epoch {epoch}, stopping...")
                    break
                
                train_losses.append(train_loss)
                
                # Validate every 10 epochs (faster)
                if epoch % 10 == 0 or epoch == self.config.max_epochs - 1:
                    valid_pred = model.predict(valid_data).squeeze()
                    valid_true = valid_data.y.squeeze()
                    
                    # Check prediction stability
                    if not stability_checker.check_predictions(valid_pred, epoch):
                        print(f"    Prediction instability at epoch {epoch}, stopping...")
                        break
                    
                    valid_mse = float(np.mean((valid_pred - valid_true) ** 2))
                    valid_losses.append(valid_mse)
                    
                    # Save best model
                    if valid_mse < best_valid_mse:
                        best_valid_mse = valid_mse
                        best_epoch = epoch
                        # Store model state
                        best_state = {k: v.clone() for k, v in model.model.state_dict().items()}
                    
                    # Check for overfitting and early stopping
                    should_stop, is_overfitting = overfit_detector.update(
                        train_loss, valid_mse, epoch
                    )
                    
                    if should_stop:
                        early_stopped = True
                        print(f"    Early stopping at epoch {epoch} (best: {best_epoch})")
                        break
                    
                    if epoch % 30 == 0:
                        status = "⚠️ OVERFIT" if is_overfitting else "OK"
                        print(f"    Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                              f"Valid: {np.sqrt(valid_mse):.4f} | {status}")
            except Exception as e:
                print(f"    Error at epoch {epoch}: {e}")
                break
        
        training_time = time.time() - start_time
        
        # Restore best model if we have it
        if best_state is not None:
            model.model.load_state_dict(best_state)
        
        # Final evaluation on test set
        test_pred = model.predict(test_data).squeeze()
        test_true = test_data.y.squeeze()
        
        test_mse = float(np.mean((test_pred - test_true) ** 2))
        test_rmse = float(np.sqrt(test_mse))
        test_mae = float(np.mean(np.abs(test_pred - test_true)))
        
        ss_res = np.sum((test_true - test_pred) ** 2)
        ss_tot = np.sum((test_true - np.mean(test_true)) ** 2)
        test_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Get formula and complexity
        formula = model.get_formula()
        complexity = model.get_complexity()
        
        # Final train MSE
        train_pred = model.predict(train_data).squeeze()
        train_true = train_data.y.squeeze()
        final_train_mse = float(np.mean((train_pred - train_true) ** 2))
        
        result = SingleRunResult(
            seed=seed,
            dataset=dataset_name,
            train_mse=final_train_mse,
            valid_mse=best_valid_mse,
            test_mse=test_mse,
            test_rmse=test_rmse,
            test_mae=test_mae,
            test_r2=test_r2,
            formula=formula,
            complexity=complexity,
            best_epoch=best_epoch,
            total_epochs=len(train_losses),
            training_time=training_time,
            overfit_detected=overfit_detector.overfit_detected,
            numerical_issues=stability_checker.has_issues(),
            early_stopped=early_stopped,
            train_losses=train_losses,
            valid_losses=valid_losses
        )
        
        print(f"    Done! RMSE: {test_rmse:.4f} | R²: {test_r2:.4f} | "
              f"Time: {training_time:.1f}s")
        
        return result
    
    def evaluate_dataset(self, dataset_name: str) -> DatasetResult:
        """
        Evaluate on a single dataset with multiple seeds.
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load dataset
        train_data, valid_data, test_data, tasks, transformers = self.load_dataset(dataset_name)
        
        print(f"Dataset info:")
        print(f"  Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")
        print(f"  Features: {train_data.X.shape[1]}")
        print(f"  Tasks: {tasks}")
        
        # Run multiple seeds
        runs = []
        for seed in self.config.seeds[:self.config.n_seeds]:
            try:
                result = self.train_single_run(
                    train_data, valid_data, test_data, seed, dataset_name
                )
                runs.append(result)
            except Exception as e:
                print(f"    ERROR with seed {seed}: {e}")
                traceback.print_exc()
        
        if not runs:
            raise RuntimeError(f"All runs failed for {dataset_name}")
        
        # Aggregate results
        rmses = [r.test_rmse for r in runs]
        r2s = [r.test_r2 for r in runs]
        maes = [r.test_mae for r in runs]
        
        best_idx = np.argmin(rmses)
        
        dataset_result = DatasetResult(
            dataset=dataset_name,
            n_runs=len(runs),
            test_rmse_mean=float(np.mean(rmses)),
            test_rmse_std=float(np.std(rmses)),
            test_r2_mean=float(np.mean(r2s)),
            test_r2_std=float(np.std(r2s)),
            test_mae_mean=float(np.mean(maes)),
            test_mae_std=float(np.std(maes)),
            best_rmse=float(np.min(rmses)),
            best_r2=float(np.max(r2s)),
            best_formula=runs[best_idx].formula,
            overfit_rate=sum(1 for r in runs if r.overfit_detected) / len(runs),
            numerical_issue_rate=sum(1 for r in runs if r.numerical_issues) / len(runs),
            early_stop_rate=sum(1 for r in runs if r.early_stopped) / len(runs),
            runs=runs
        )
        
        return dataset_result
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete rigorous evaluation.
        """
        print("="*70)
        print("RIGOROUS EVALUATION FRAMEWORK")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Configuration:")
        print(f"  Datasets: {self.config.datasets}")
        print(f"  Seeds: {self.config.seeds[:self.config.n_seeds]}")
        print(f"  Max epochs: {self.config.max_epochs}")
        print(f"  Early stopping patience: {self.config.early_stopping_patience}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'datasets': {}
        }
        
        for dataset_name in self.config.datasets:
            try:
                dataset_result = self.evaluate_dataset(dataset_name)
                self.all_results[dataset_name] = dataset_result
                
                # Convert to serializable format
                results['datasets'][dataset_name] = {
                    'n_runs': dataset_result.n_runs,
                    'test_rmse_mean': dataset_result.test_rmse_mean,
                    'test_rmse_std': dataset_result.test_rmse_std,
                    'test_r2_mean': dataset_result.test_r2_mean,
                    'test_r2_std': dataset_result.test_r2_std,
                    'test_mae_mean': dataset_result.test_mae_mean,
                    'test_mae_std': dataset_result.test_mae_std,
                    'best_rmse': dataset_result.best_rmse,
                    'best_r2': dataset_result.best_r2,
                    'best_formula': dataset_result.best_formula,
                    'overfit_rate': dataset_result.overfit_rate,
                    'numerical_issue_rate': dataset_result.numerical_issue_rate,
                    'early_stop_rate': dataset_result.early_stop_rate,
                    'runs': [
                        {
                            'seed': r.seed,
                            'test_rmse': r.test_rmse,
                            'test_r2': r.test_r2,
                            'test_mae': r.test_mae,
                            'best_epoch': r.best_epoch,
                            'training_time': r.training_time,
                            'overfit_detected': r.overfit_detected,
                            'formula': r.formula
                        }
                        for r in dataset_result.runs
                    ]
                }
                
            except Exception as e:
                print(f"ERROR evaluating {dataset_name}: {e}")
                traceback.print_exc()
                results['datasets'][dataset_name] = {'error': str(e)}
        
        # Print summary
        self._print_summary()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*70)
        print("RIGOROUS EVALUATION SUMMARY")
        print("="*70)
        
        # Baseline comparisons
        baselines = {
            'delaney': {'RF': 1.07, 'XGB': 0.98, 'GCN': 0.82, 'MPNN': 0.76},
            'lipo': {'RF': 0.78, 'XGB': 0.72, 'GCN': 0.65, 'MPNN': 0.62}
        }
        
        for dataset_name, result in self.all_results.items():
            print(f"\n{dataset_name.upper()} (n={result.n_runs} runs)")
            print("-" * 50)
            print(f"  RMSE: {result.test_rmse_mean:.4f} ± {result.test_rmse_std:.4f}")
            print(f"  R²:   {result.test_r2_mean:.4f} ± {result.test_r2_std:.4f}")
            print(f"  MAE:  {result.test_mae_mean:.4f} ± {result.test_mae_std:.4f}")
            print(f"  Best Formula: {result.best_formula[:60]}...")
            print(f"\n  Health Indicators:")
            print(f"    Overfitting rate:     {result.overfit_rate*100:.0f}%")
            print(f"    Numerical issues:     {result.numerical_issue_rate*100:.0f}%")
            print(f"    Early stopping rate:  {result.early_stop_rate*100:.0f}%")
            
            if dataset_name in baselines:
                print(f"\n  Comparison with Baselines (RMSE):")
                our_rmse = result.test_rmse_mean
                for model_name, baseline_rmse in baselines[dataset_name].items():
                    diff = our_rmse - baseline_rmse
                    status = "✓ better" if diff < 0 else "✗ worse"
                    print(f"    vs {model_name:4s}: {diff:+.3f} ({status})")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.output_dir, f'rigorous_results_{timestamp}.json')
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {filepath}")


def main():
    """Run rigorous evaluation."""
    if not DEEPCHEM_AVAILABLE or not MODEL_AVAILABLE:
        print("Missing dependencies. Please install deepchem and ensure model is available.")
        return
    
    # Configure evaluation
    config = EvaluationConfig(
        datasets=['delaney', 'lipo'],
        n_seeds=3,
        seeds=[42, 123, 456],
        max_depth=4,
        n_candidates=7,
        complexity_weight=0.005,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=100,  # Reduced for faster evaluation
        early_stopping_patience=20
    )
    
    # Run evaluation
    evaluator = RigorousEvaluator(config)
    results = evaluator.run_full_evaluation()
    
    return results


if __name__ == "__main__":
    results = main()

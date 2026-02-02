"""
MoleculeNet Evaluation Script for Enhanced Symbolic Regression Model
====================================================================

This script evaluates the enhanced symbolic regression model on MoleculeNet
benchmark datasets to test compatibility with real-world molecular property
prediction tasks.

MoleculeNet Datasets Used:
    - Delaney (ESOL): Water solubility prediction (1128 compounds)
    - FreeSolv: Hydration free energy (642 compounds)
    - Lipo: Lipophilicity prediction (4200 compounds)

These are regression tasks suitable for symbolic regression.

Author: GSoC Symbolic Regression Project
Date: 2026-02-02
"""

import sys
import os
import time
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

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
    from deepchem.data import NumpyDataset
    from deepchem.metrics import Metric
    from deepchem.molnet import load_delaney, load_lipo
    # Try importing optional datasets
    try:
        from deepchem.molnet import load_freesolv
        FREESOLV_AVAILABLE = True
    except ImportError:
        FREESOLV_AVAILABLE = False
        load_freesolv = None
    DEEPCHEM_AVAILABLE = True
    print(f"✓ DeepChem version: {dc.__version__}")
except ImportError as e:
    print(f"✗ DeepChem not available: {e}")
    DEEPCHEM_AVAILABLE = False
    FREESOLV_AVAILABLE = False

# Import our enhanced model
try:
    from models.symbolic_regressor import (
        SymbolicRegressorModel, 
        DPSymbolicRegressorModel,
        SymbolicRegressorModule
    )
    SYMBOLIC_MODEL_AVAILABLE = True
    print("✓ Symbolic Regression Model loaded")
except ImportError as e:
    print(f"✗ Symbolic Regression Model not available: {e}")
    SYMBOLIC_MODEL_AVAILABLE = False

# Import hybrid trainer
try:
    from hybrid_optimization.hybrid_trainer import HybridTrainer, HybridConfig
    HYBRID_AVAILABLE = True
    print("✓ Hybrid Optimization loaded")
except ImportError as e:
    print(f"✗ Hybrid Optimization not available: {e}")
    HYBRID_AVAILABLE = False


class MoleculeNetEvaluator:
    """
    Evaluator for testing symbolic regression on MoleculeNet datasets.
    
    This class handles:
        1. Loading and preprocessing MoleculeNet datasets
        2. Training symbolic regression models
        3. Computing evaluation metrics (RMSE, R², MAE)
        4. Comparing with baseline results
        5. Generating reports
    """
    
    # Baseline results from MoleculeNet paper and DeepChem benchmarks
    # These are typical results for comparison
    BASELINE_RESULTS = {
        'delaney': {
            'Random Forest': {'rmse': 1.07, 'r2': 0.80},
            'XGBoost': {'rmse': 0.98, 'r2': 0.83},
            'Graph Conv': {'rmse': 0.82, 'r2': 0.87},
            'MPNN': {'rmse': 0.76, 'r2': 0.90},
            'Linear Regression': {'rmse': 1.32, 'r2': 0.69},
        },
        'freesolv': {
            'Random Forest': {'rmse': 1.45, 'r2': 0.85},
            'XGBoost': {'rmse': 1.35, 'r2': 0.87},
            'Graph Conv': {'rmse': 1.15, 'r2': 0.91},
            'MPNN': {'rmse': 1.05, 'r2': 0.93},
            'Linear Regression': {'rmse': 1.82, 'r2': 0.78},
        },
        'lipo': {
            'Random Forest': {'rmse': 0.78, 'r2': 0.65},
            'XGBoost': {'rmse': 0.72, 'r2': 0.70},
            'Graph Conv': {'rmse': 0.65, 'r2': 0.75},
            'MPNN': {'rmse': 0.62, 'r2': 0.78},
            'Linear Regression': {'rmse': 0.95, 'r2': 0.48},
        }
    }
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save results and reports
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")
    
    def load_dataset(
        self, 
        dataset_name: str,
        featurizer: str = 'ECFP',
        splitter: str = 'scaffold'
    ) -> Tuple[Any, Any, Any, List[str]]:
        """
        Load a MoleculeNet dataset with specified featurization.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset ('delaney', 'freesolv', 'lipo')
        featurizer : str
            Featurizer to use ('ECFP', 'GraphConv', 'Weave')
        splitter : str
            Data splitting strategy ('scaffold', 'random')
            
        Returns
        -------
        tuple
            (train_dataset, valid_dataset, test_dataset, task_names)
        """
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name} dataset with {featurizer} features...")
        print(f"{'='*60}")
        
        loaders = {
            'delaney': load_delaney,
            'lipo': load_lipo
        }
        
        # Add freesolv only if available
        if FREESOLV_AVAILABLE:
            loaders['freesolv'] = load_freesolv
        
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        tasks, datasets, transformers = loaders[dataset_name](
            featurizer=featurizer,
            splitter=splitter,
            transformers=['normalization']
        )
        
        train, valid, test = datasets
        
        print(f"  Tasks: {tasks}")
        print(f"  Train samples: {len(train)}")
        print(f"  Valid samples: {len(valid)}")
        print(f"  Test samples: {len(test)}")
        print(f"  Features shape: {train.X.shape}")
        
        return train, valid, test, tasks
    
    def train_symbolic_model(
        self,
        train_dataset,
        valid_dataset,
        max_depth: int = 4,
        n_candidates: int = 7,
        complexity_weight: float = 0.005,
        learning_rate: float = 0.01,
        n_epochs: int = 200,
        batch_size: int = 64,
        use_dp: bool = True,
        verbose: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the enhanced symbolic regression model.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training data
        valid_dataset : Dataset
            Validation data for early stopping
        max_depth : int
            Maximum expression tree depth
        n_candidates : int
            Number of candidate expressions
        complexity_weight : float
            Weight for complexity penalty
        learning_rate : float
            Learning rate for optimization
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        use_dp : bool
            Whether to use DP-optimized model
        verbose : bool
            Print training progress
            
        Returns
        -------
        tuple
            (trained_model, training_info)
        """
        n_features = train_dataset.X.shape[1]
        
        print(f"\n{'='*60}")
        print("Training Enhanced Symbolic Regression Model")
        print(f"{'='*60}")
        print(f"  Features: {n_features}")
        print(f"  Max depth: {max_depth}")
        print(f"  Candidates: {n_candidates}")
        print(f"  Complexity weight: {complexity_weight}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Using DP optimization: {use_dp}")
        
        # Create model
        ModelClass = DPSymbolicRegressorModel if use_dp else SymbolicRegressorModel
        model = ModelClass(
            n_features=n_features,
            max_depth=max_depth,
            n_candidates=n_candidates,
            complexity_weight=complexity_weight,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=self.device
        )
        
        # Training with progress tracking
        start_time = time.time()
        training_losses = []
        best_valid_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Train for one epoch
            loss = model.fit(train_dataset, nb_epoch=1, deterministic=False)
            training_losses.append(loss)
            
            # Validate
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                valid_pred = model.predict(valid_dataset).squeeze()
                valid_true = valid_dataset.y.squeeze()
                valid_mse = np.mean((valid_pred - valid_true) ** 2)
                valid_rmse = np.sqrt(valid_mse)
                
                if valid_mse < best_valid_loss:
                    best_valid_loss = valid_mse
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and epoch % 20 == 0:
                    print(f"  Epoch {epoch:4d} | Train Loss: {loss:.6f} | "
                          f"Valid RMSE: {valid_rmse:.4f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        training_time = time.time() - start_time
        
        # Collect training info
        training_info = {
            'n_epochs_actual': epoch + 1,
            'training_time': training_time,
            'final_train_loss': training_losses[-1] if training_losses else None,
            'best_valid_mse': best_valid_loss,
            'training_losses': training_losses,
            'hyperparameters': {
                'n_features': n_features,
                'max_depth': max_depth,
                'n_candidates': n_candidates,
                'complexity_weight': complexity_weight,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
        }
        
        print(f"\n  Training completed in {training_time:.1f}s")
        
        return model, training_info
    
    def evaluate_model(
        self,
        model,
        test_dataset,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Parameters
        ----------
        model : SymbolicRegressorModel
            Trained model
        test_dataset : Dataset
            Test data
        dataset_name : str
            Name of dataset (for baseline comparison)
            
        Returns
        -------
        dict
            Evaluation metrics and formula
        """
        print(f"\n{'='*60}")
        print("Evaluating Model on Test Set")
        print(f"{'='*60}")
        
        # Get predictions
        start_time = time.time()
        predictions = model.predict(test_dataset).squeeze()
        inference_time = time.time() - start_time
        
        y_true = test_dataset.y.squeeze()
        
        # Compute metrics
        mse = np.mean((predictions - y_true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_true))
        
        # R² score
        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # Get formula
        formula = model.get_formula()
        complexity = model.get_complexity()
        
        results = {
            'rmse': float(rmse),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'formula': formula,
            'complexity': float(complexity),
            'inference_time': inference_time,
            'n_test_samples': len(test_dataset)
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Complexity: {complexity:.2f}")
        print(f"  Inference time: {inference_time*1000:.1f}ms")
        print(f"\n  Discovered Formula:")
        print(f"    {formula}")
        
        # Compare with baselines
        if dataset_name in self.BASELINE_RESULTS:
            print(f"\n  Comparison with Baselines:")
            baselines = self.BASELINE_RESULTS[dataset_name]
            results['baseline_comparison'] = {}
            
            for model_name, baseline in baselines.items():
                rmse_diff = rmse - baseline['rmse']
                r2_diff = r2 - baseline['r2']
                status_rmse = "✓ better" if rmse_diff < 0 else "✗ worse"
                status_r2 = "✓ better" if r2_diff > 0 else "✗ worse"
                
                print(f"    vs {model_name:20s}: RMSE {rmse_diff:+.3f} ({status_rmse}), "
                      f"R² {r2_diff:+.3f} ({status_r2})")
                
                results['baseline_comparison'][model_name] = {
                    'rmse_diff': float(rmse_diff),
                    'r2_diff': float(r2_diff)
                }
        
        return results
    
    def run_full_evaluation(
        self,
        datasets: List[str] = ['delaney', 'freesolv', 'lipo'],
        featurizer: str = 'ECFP',
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Run full evaluation on multiple MoleculeNet datasets.
        
        Parameters
        ----------
        datasets : list of str
            Datasets to evaluate on
        featurizer : str
            Featurizer to use
        **training_kwargs
            Additional arguments for training
            
        Returns
        -------
        dict
            All results
        """
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'featurizer': featurizer,
            'training_config': training_kwargs,
            'datasets': {}
        }
        
        for dataset_name in datasets:
            print(f"\n{'#'*70}")
            print(f"# Processing: {dataset_name.upper()}")
            print(f"{'#'*70}")
            
            try:
                # Load dataset
                train, valid, test, tasks = self.load_dataset(
                    dataset_name, featurizer=featurizer
                )
                
                # Train model
                model, training_info = self.train_symbolic_model(
                    train, valid, **training_kwargs
                )
                
                # Evaluate
                eval_results = self.evaluate_model(model, test, dataset_name)
                
                # Store results
                all_results['datasets'][dataset_name] = {
                    'training_info': training_info,
                    'evaluation': eval_results,
                    'tasks': tasks
                }
                
            except Exception as e:
                print(f"  ✗ Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results['datasets'][dataset_name] = {'error': str(e)}
        
        # Save results
        self._save_results(all_results)
        
        # Generate summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        # Convert non-serializable items
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj
        
        serializable = make_serializable(results)
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.output_dir, f'results_{timestamp}.json')
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        for dataset_name, data in results['datasets'].items():
            if 'error' in data:
                print(f"\n{dataset_name}: ERROR - {data['error']}")
                continue
            
            eval_data = data['evaluation']
            print(f"\n{dataset_name.upper()}:")
            print(f"  RMSE: {eval_data['rmse']:.4f}")
            print(f"  R²:   {eval_data['r2']:.4f}")
            print(f"  MAE:  {eval_data['mae']:.4f}")
            print(f"  Formula complexity: {eval_data['complexity']:.2f}")
            
            # Show formula (truncated if long)
            formula = eval_data['formula']
            if len(formula) > 100:
                formula = formula[:100] + "..."
            print(f"  Formula: {formula}")


def main():
    """Main evaluation function."""
    print("="*70)
    print("MoleculeNet Evaluation for Enhanced Symbolic Regression")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not DEEPCHEM_AVAILABLE:
        print("\n✗ DeepChem is required for MoleculeNet evaluation")
        print("  Install with: pip install deepchem")
        return
    
    if not SYMBOLIC_MODEL_AVAILABLE:
        print("\n✗ Symbolic Regression Model not available")
        return
    
    # Create evaluator
    evaluator = MoleculeNetEvaluator()
    
    # Run evaluation
    # Start with Delaney as it's smaller and good for testing
    results = evaluator.run_full_evaluation(
        datasets=['delaney', 'lipo'],  # Test on multiple datasets
        featurizer='ECFP',
        max_depth=4,
        n_candidates=7,
        complexity_weight=0.005,
        learning_rate=0.01,
        n_epochs=150,
        batch_size=64,
        use_dp=True,
        verbose=True
    )
    
    return results


if __name__ == "__main__":
    results = main()

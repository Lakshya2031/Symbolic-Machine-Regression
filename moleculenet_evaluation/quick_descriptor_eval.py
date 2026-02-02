"""
Quick Descriptor-Based Evaluation
=================================
Faster evaluation script for DP-enhanced symbolic regression with RDKit descriptors.
"""

import sys
import os
import time
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'symbolic_regression', 'src'))

import torch

# DeepChem imports
import deepchem as dc
from deepchem.data import NumpyDataset
print(f"DeepChem: {dc.__version__}")

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen
print("RDKit: OK")

# Model import
from models.symbolic_regressor import SymbolicRegressorModel

# Sklearn for baselines
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# DESCRIPTOR FEATURIZER (SIMPLIFIED)
# =============================================================================

DESCRIPTOR_FUNCS = {
    'MolWt': Descriptors.MolWt,
    'LogP': Crippen.MolLogP,
    'TPSA': rdMolDescriptors.CalcTPSA,
    'NumHDonors': Lipinski.NumHDonors,
    'NumHAcceptors': Lipinski.NumHAcceptors,
    'NumRotatableBonds': Lipinski.NumRotatableBonds,
    'NumAromaticRings': Lipinski.NumAromaticRings,
    'NumSaturatedRings': Lipinski.NumSaturatedRings,
    'RingCount': Lipinski.RingCount,
    'FractionCSP3': Lipinski.FractionCSP3,
    'NumHeteroatoms': Lipinski.NumHeteroatoms,
    'HeavyAtomCount': Lipinski.HeavyAtomCount,
    'NOCount': Lipinski.NOCount,
    'NHOHCount': Lipinski.NHOHCount,
    'BertzCT': Descriptors.BertzCT,
    'Chi0': Descriptors.Chi0,
    'Kappa1': Descriptors.Kappa1,
    'Kappa2': Descriptors.Kappa2,
    'LabuteASA': Descriptors.LabuteASA,
}

DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCS.keys())
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)

def featurize_smiles(smiles: str) -> np.ndarray:
    """Compute descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_DESCRIPTORS, dtype=np.float32)
    
    features = []
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            val = func(mol)
            if val is None or not np.isfinite(val):
                val = 0.0
            features.append(val)
        except:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def load_dataset_with_descriptors(dataset_name: str):
    """Load a MoleculeNet dataset with descriptor featurization."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} with {N_DESCRIPTORS} RDKit Descriptors")
    print(f"{'='*60}")
    
    # Load raw data
    if dataset_name == 'delaney':
        from deepchem.molnet import load_delaney
        tasks, datasets, _ = load_delaney(featurizer='Raw', splitter='scaffold')
    elif dataset_name == 'lipo':
        from deepchem.molnet import load_lipo
        tasks, datasets, _ = load_lipo(featurizer='Raw', splitter='scaffold')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_ds, valid_ds, test_ds = datasets
    
    # Featurize
    scaler = StandardScaler()
    
    print(f"  Featurizing {len(train_ds.ids)} train molecules...")
    X_train = np.array([featurize_smiles(s) for s in train_ds.ids])
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    
    print(f"  Featurizing {len(valid_ds.ids)} valid molecules...")
    X_valid = np.array([featurize_smiles(s) for s in valid_ds.ids])
    X_valid = np.nan_to_num(X_valid, nan=0.0)
    X_valid = scaler.transform(X_valid).astype(np.float32)
    
    print(f"  Featurizing {len(test_ds.ids)} test molecules...")
    X_test = np.array([featurize_smiles(s) for s in test_ds.ids])
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    y_train = train_ds.y.squeeze().astype(np.float32)
    y_valid = valid_ds.y.squeeze().astype(np.float32)
    y_test = test_ds.y.squeeze().astype(np.float32)
    
    print(f"  ✓ Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_symbolic_regression(X_train, y_train, X_valid, y_valid, max_epochs=100, patience=20):
    """Train symbolic regression with early stopping."""
    train_ds = NumpyDataset(X=X_train, y=y_train.reshape(-1, 1))
    valid_ds = NumpyDataset(X=X_valid, y=y_valid.reshape(-1, 1))
    
    model = SymbolicRegressorModel(
        n_features=N_DESCRIPTORS,
        max_depth=3,  # Simple trees
        n_candidates=5,
        complexity_weight=0.1,  # Strong regularization
        learning_rate=0.01,
        batch_size=32
    )
    
    best_valid_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    print(f"\n  Training (max_epochs={max_epochs}, patience={patience})...")
    
    for epoch in range(max_epochs):
        # Train
        model.fit(train_ds, nb_epoch=1, deterministic=False)
        
        # Evaluate
        train_pred = model.predict(train_ds).squeeze()
        valid_pred = model.predict(valid_ds).squeeze()
        
        train_r2 = r2_score(y_train, train_pred)
        valid_r2 = r2_score(y_valid, valid_pred)
        valid_mse = mean_squared_error(y_valid, valid_pred)
        
        if valid_mse < best_valid_loss - 0.0001:
            best_valid_loss = valid_mse
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.model.state_dict().items()}
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or patience_counter >= patience:
            print(f"    Epoch {epoch:3d} | Train R²: {train_r2:.4f} | Valid R²: {valid_r2:.4f}")
        
        if patience_counter >= patience:
            print(f"    → Early stop at epoch {epoch} (best: {best_epoch})")
            break
    
    # Restore best
    if best_state:
        model.model.load_state_dict(best_state)
    
    return model


def main():
    """Run evaluation."""
    print("="*70)
    print("DESCRIPTOR-BASED SYMBOLIC REGRESSION EVALUATION")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Features: {N_DESCRIPTORS} RDKit physicochemical descriptors")
    print(f"Descriptors: {DESCRIPTOR_NAMES[:5]}...")
    
    results = {}
    
    for dataset_name in ['delaney', 'lipo']:
        try:
            # Load data
            X_train, y_train, X_valid, y_valid, X_test, y_test = \
                load_dataset_with_descriptors(dataset_name)
            
            # ---- BASELINES ----
            print(f"\n  BASELINES:")
            
            # Linear Regression
            lr = Ridge(alpha=1.0)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            lr_r2 = r2_score(y_test, lr_pred)
            print(f"    Linear Regression: RMSE={lr_rmse:.4f}, R²={lr_r2:.4f}")
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)
            print(f"    Random Forest:     RMSE={rf_rmse:.4f}, R²={rf_r2:.4f}")
            
            # ---- SYMBOLIC REGRESSION ----
            print(f"\n  SYMBOLIC REGRESSION (DP-enhanced):")
            model = train_symbolic_regression(X_train, y_train, X_valid, y_valid)
            
            # Evaluate
            test_ds = NumpyDataset(X=X_test, y=y_test.reshape(-1, 1))
            test_pred = model.predict(test_ds).squeeze()
            sr_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            sr_r2 = r2_score(y_test, test_pred)
            
            # Get formula
            formula = model.get_formula(var_names=DESCRIPTOR_NAMES)
            complexity = model.get_complexity()
            
            print(f"\n  RESULTS - {dataset_name.upper()}:")
            print(f"  {'-'*50}")
            print(f"    Test RMSE: {sr_rmse:.4f}")
            print(f"    Test R²:   {sr_r2:.4f}")
            print(f"    Complexity: {complexity:.2f}")
            print(f"    Formula: {formula}")
            
            # Compare
            print(f"\n  COMPARISON:")
            print(f"    {'Model':<25} {'RMSE':<10} {'R²':<10}")
            print(f"    {'-'*45}")
            print(f"    {'Linear Regression':<25} {lr_rmse:<10.4f} {lr_r2:<10.4f}")
            print(f"    {'Random Forest':<25} {rf_rmse:<10.4f} {rf_r2:<10.4f}")
            print(f"    {'Symbolic Regression':<25} {sr_rmse:<10.4f} {sr_r2:<10.4f}")
            
            # Store
            results[dataset_name] = {
                'linear_regression': {'rmse': float(lr_rmse), 'r2': float(lr_r2)},
                'random_forest': {'rmse': float(rf_rmse), 'r2': float(rf_r2)},
                'symbolic_regression': {
                    'rmse': float(sr_rmse),
                    'r2': float(sr_r2),
                    'complexity': float(complexity),
                    'formula': formula
                }
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== ANALYSIS ==========
    print("\n" + "="*70)
    print("ANALYSIS: DESCRIPTORS vs ECFP")
    print("="*70)
    
    print("""
WHY DESCRIPTORS WORK BETTER FOR SYMBOLIC REGRESSION:
─────────────────────────────────────────────────────
1. DIMENSIONALITY: 19 features vs 1024 - tractable search space
2. CONTINUOUS VALUES: Gradients flow properly (not 0/1 sparse)
3. INTERPRETABLE: Formula uses MolWt, LogP, TPSA (meaningful!)
4. DP BENEFITS: Same subexpressions reused → cache hits

COMPARISON WITH PREVIOUS ECFP RESULTS:
──────────────────────────────────────
| Metric       | ECFP (1024)  | Descriptors (19) | Change    |
|--------------|--------------|------------------|-----------|""")
    
    if 'delaney' in results:
        sr_d = results['delaney']['symbolic_regression']
        print(f"| Delaney RMSE | 0.745        | {sr_d['rmse']:<16.4f} | {'↓ better' if sr_d['rmse'] < 0.745 else '↑ worse':<9} |")
        print(f"| Delaney R²   | 0.47         | {sr_d['r2']:<16.4f} | {'↑ better' if sr_d['r2'] > 0.47 else '↓ worse':<9} |")
    
    if 'lipo' in results:
        sr_l = results['lipo']['symbolic_regression']
        print(f"| Lipo RMSE    | 0.875        | {sr_l['rmse']:<16.4f} | {'↓ better' if sr_l['rmse'] < 0.875 else '↑ worse':<9} |")
        print(f"| Lipo R²      | 0.08         | {sr_l['r2']:<16.4f} | {'↑ better' if sr_l['r2'] > 0.08 else '↓ worse':<9} |")
    
    print("""
──────────────────────────────────────────────────────────────────────

INTERPRETABILITY EXAMPLE:
─────────────────────────""")
    
    for ds, data in results.items():
        if 'symbolic_regression' in data:
            print(f"\n{ds.upper()} Formula:")
            print(f"  {data['symbolic_regression']['formula']}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'descriptor_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'descriptor_eval_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {filepath}")
    
    return results


if __name__ == "__main__":
    main()

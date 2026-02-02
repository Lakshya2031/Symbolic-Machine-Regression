"""
Adaptive Descriptor-Based Symbolic Regression Evaluation
=========================================================

Key Improvement: Hyperparameters are ADAPTIVE and LEARNED per dataset
through validation-based tuning, not fixed across all datasets.

This is more scientifically rigorous because:
1. Different datasets have different characteristics (size, noise, complexity)
2. Optimal regularization depends on dataset size
3. Learning rate should adapt to feature scale
4. Complexity penalty should match inherent problem complexity

Author: GSoC Symbolic Regression Project
Date: February 2, 2026
"""

import sys
import os
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
from itertools import product

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'symbolic_regression', 'src'))

import torch
import torch.nn as nn
import deepchem as dc
from deepchem.data import NumpyDataset

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print(f"DeepChem: {dc.__version__}")
print(f"PyTorch: {torch.__version__}")


# =============================================================================
# DESCRIPTOR CONFIGURATION
# =============================================================================

DESCRIPTOR_FUNCS = {
    'MolWt': Descriptors.MolWt,
    'LogP': Crippen.MolLogP,
    'TPSA': rdMolDescriptors.CalcTPSA,
    'NumHDonors': Lipinski.NumHDonors,
    'NumHAcceptors': Lipinski.NumHAcceptors,
    'NumRotatableBonds': Lipinski.NumRotatableBonds,
    'NumAromaticRings': Lipinski.NumAromaticRings,
    'RingCount': Lipinski.RingCount,
    'FractionCSP3': Lipinski.FractionCSP3,
    'HeavyAtomCount': Lipinski.HeavyAtomCount,
    'NOCount': Lipinski.NOCount,
    'NHOHCount': Lipinski.NHOHCount,
    'Chi0': Descriptors.Chi0,
    'Kappa1': Descriptors.Kappa1,
    'LabuteASA': Descriptors.LabuteASA,
}

DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCS.keys())
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def featurize_molecule(smiles: str) -> np.ndarray:
    """Compute descriptors for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_DESCRIPTORS, dtype=np.float32)
    
    features = []
    for func in DESCRIPTOR_FUNCS.values():
        try:
            val = func(mol)
            if val is None or not np.isfinite(val):
                val = 0.0
            features.append(float(val))
        except:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# DATASET LOADER WITH CHARACTERISTICS ANALYSIS
# =============================================================================

def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """Load dataset and analyze its characteristics for adaptive hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} with Characteristic Analysis")
    print(f"{'='*60}")
    
    if dataset_name == 'delaney':
        from deepchem.molnet import load_delaney
        tasks, datasets, _ = load_delaney(featurizer='Raw', splitter='scaffold')
    else:
        from deepchem.molnet import load_lipo
        tasks, datasets, _ = load_lipo(featurizer='Raw', splitter='scaffold')
    
    train_ds, valid_ds, test_ds = datasets
    scaler = StandardScaler()
    
    # Featurize
    print(f"  Featurizing {len(train_ds.ids)} train molecules...")
    X_train = np.array([featurize_molecule(s) for s in train_ds.ids])
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    
    print(f"  Featurizing {len(valid_ds.ids)} valid molecules...")
    X_valid = np.array([featurize_molecule(s) for s in valid_ds.ids])
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
    X_valid = scaler.transform(X_valid).astype(np.float32)
    
    print(f"  Featurizing {len(test_ds.ids)} test molecules...")
    X_test = np.array([featurize_molecule(s) for s in test_ds.ids])
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    y_train = train_ds.y.squeeze().astype(np.float32)
    y_valid = valid_ds.y.squeeze().astype(np.float32)
    y_test = test_ds.y.squeeze().astype(np.float32)
    
    # ========== ANALYZE DATASET CHARACTERISTICS ==========
    characteristics = {
        'n_train': len(y_train),
        'n_valid': len(y_valid),
        'n_test': len(y_test),
        'n_features': N_DESCRIPTORS,
        'y_mean': float(y_train.mean()),
        'y_std': float(y_train.std()),
        'y_range': float(y_train.max() - y_train.min()),
        'feature_variance': float(X_train.var(axis=0).mean()),
        'samples_per_feature': len(y_train) / N_DESCRIPTORS,
    }
    
    # Estimate problem complexity using linear baseline
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    lr_r2 = r2_score(y_valid, lr.predict(X_valid))
    characteristics['linear_baseline_r2'] = float(lr_r2)
    
    # Estimate if problem is "easy" (linear works well) or "hard"
    characteristics['problem_difficulty'] = 'easy' if lr_r2 > 0.6 else ('medium' if lr_r2 > 0.3 else 'hard')
    
    print(f"\n  Dataset Characteristics:")
    print(f"    Samples: {characteristics['n_train']} train, {characteristics['n_valid']} valid")
    print(f"    Y range: [{y_train.min():.2f}, {y_train.max():.2f}], std={characteristics['y_std']:.2f}")
    print(f"    Samples per feature: {characteristics['samples_per_feature']:.1f}")
    print(f"    Linear baseline R²: {lr_r2:.4f} → Problem difficulty: {characteristics['problem_difficulty']}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_valid': X_valid, 'y_valid': y_valid,
        'X_test': X_test, 'y_test': y_test,
        'characteristics': characteristics,
        'scaler': scaler
    }


# =============================================================================
# ADAPTIVE HYPERPARAMETER TUNER
# =============================================================================

class AdaptiveHyperparameterTuner:
    """
    Learns optimal hyperparameters for each dataset based on:
    1. Dataset size (more data → can use more complex models)
    2. Problem difficulty (harder → need more regularization)
    3. Validation performance feedback
    """
    
    def __init__(self, characteristics: Dict[str, Any]):
        self.characteristics = characteristics
        self.best_params = None
        self.search_history = []
    
    def suggest_initial_params(self) -> Dict[str, Any]:
        """Suggest initial hyperparameters based on dataset characteristics."""
        c = self.characteristics
        
        # Adaptive learning rate based on dataset size
        # Larger datasets → can use smaller LR for better convergence
        if c['n_train'] > 2000:
            lr = 0.005
        elif c['n_train'] > 500:
            lr = 0.01
        else:
            lr = 0.02
        
        # Adaptive complexity penalty based on problem difficulty
        if c['problem_difficulty'] == 'easy':
            # Easy problem: use simpler models, higher penalty
            complexity_weight = 0.05
            max_depth = 2
        elif c['problem_difficulty'] == 'medium':
            complexity_weight = 0.02
            max_depth = 3
        else:
            # Hard problem: allow more complexity, lower penalty
            complexity_weight = 0.01
            max_depth = 3
        
        # Adaptive L1 regularization based on samples per feature
        if c['samples_per_feature'] < 30:
            l1_reg = 0.01  # More regularization when data is scarce
        elif c['samples_per_feature'] < 100:
            l1_reg = 0.005
        else:
            l1_reg = 0.001
        
        # Adaptive batch size
        batch_size = min(64, max(16, c['n_train'] // 20))
        
        # Adaptive patience (more data → can wait longer)
        patience = min(50, max(20, c['n_train'] // 50))
        
        return {
            'learning_rate': lr,
            'complexity_weight': complexity_weight,
            'max_depth': max_depth,
            'l1_reg': l1_reg,
            'batch_size': batch_size,
            'patience': patience,
            'n_candidates': 3 if c['problem_difficulty'] == 'easy' else 5
        }
    
    def get_search_space(self) -> List[Dict[str, Any]]:
        """Get hyperparameter search space adapted to dataset."""
        c = self.characteristics
        base = self.suggest_initial_params()
        
        # Create focused search around suggested params
        search_space = []
        
        # Learning rates to try
        lrs = [base['learning_rate'] * 0.5, base['learning_rate'], base['learning_rate'] * 2]
        
        # Complexity weights to try
        cws = [base['complexity_weight'] * 0.5, base['complexity_weight'], base['complexity_weight'] * 2]
        
        # Max depths to try
        depths = [max(2, base['max_depth'] - 1), base['max_depth']]
        
        for lr, cw, depth in product(lrs, cws, depths):
            params = base.copy()
            params['learning_rate'] = lr
            params['complexity_weight'] = cw
            params['max_depth'] = depth
            search_space.append(params)
        
        return search_space
    
    def tune(self, train_func, X_train, y_train, X_valid, y_valid, max_trials: int = 6):
        """Run adaptive hyperparameter tuning."""
        print(f"\n  Adaptive Hyperparameter Tuning (max {max_trials} trials)...")
        
        search_space = self.get_search_space()[:max_trials]
        
        best_score = -float('inf')
        best_params = None
        
        for i, params in enumerate(search_space):
            print(f"    Trial {i+1}/{len(search_space)}: lr={params['learning_rate']:.4f}, "
                  f"cw={params['complexity_weight']:.3f}, depth={params['max_depth']}")
            
            try:
                model, metrics = train_func(X_train, y_train, X_valid, y_valid, params, verbose=False)
                score = metrics['valid_r2']
                
                self.search_history.append({
                    'params': params.copy(),
                    'valid_r2': score,
                    'valid_rmse': metrics['valid_rmse']
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"      → New best! Valid R²: {score:.4f}")
                else:
                    print(f"      Valid R²: {score:.4f}")
                    
            except Exception as e:
                print(f"      → Failed: {e}")
                continue
        
        self.best_params = best_params if best_params else self.suggest_initial_params()
        print(f"\n  Best hyperparameters found:")
        print(f"    Learning rate: {self.best_params['learning_rate']}")
        print(f"    Complexity weight: {self.best_params['complexity_weight']}")
        print(f"    Max depth: {self.best_params['max_depth']}")
        
        return self.best_params


# =============================================================================
# SYMBOLIC REGRESSION WITH ADAPTIVE TRAINING
# =============================================================================

class AdaptiveSymbolicRegressor(nn.Module):
    """Symbolic regression model with polynomial basis."""
    
    def __init__(self, n_features: int, max_degree: int = 2, include_interactions: bool = True):
        super().__init__()
        self.n_features = n_features
        self.max_degree = max_degree
        self.include_interactions = include_interactions
        
        # Linear terms
        self.linear = nn.Linear(n_features, 1)
        
        # Quadratic terms
        if include_interactions:
            n_quad = (n_features * (n_features + 1)) // 2
            self.quad_weight = nn.Parameter(torch.zeros(n_quad))
        else:
            self.quad_weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x).squeeze(-1)
        
        if self.quad_weight is not None:
            quad_terms = []
            for i in range(self.n_features):
                for j in range(i, self.n_features):
                    quad_terms.append(x[:, i] * x[:, j])
            quad = torch.stack(quad_terms, dim=1)
            out = out + (quad * self.quad_weight).sum(dim=1)
        
        return out
    
    def get_formula(self, var_names: List[str], threshold: float = 0.01) -> str:
        """Extract interpretable formula with adaptive threshold."""
        linear_w = self.linear.weight.data.squeeze().cpu().numpy()
        bias = self.linear.bias.data.item()
        
        terms = []
        
        # Linear terms (sorted by importance)
        linear_importance = [(abs(w), i, w) for i, w in enumerate(linear_w)]
        linear_importance.sort(reverse=True)
        
        for _, i, w in linear_importance:
            if abs(w) > threshold:
                terms.append(f"{w:.3f}*{var_names[i]}")
        
        # Quadratic terms
        if self.quad_weight is not None:
            quad_w = self.quad_weight.data.cpu().numpy()
            idx = 0
            quad_terms = []
            for i in range(self.n_features):
                for j in range(i, self.n_features):
                    w = quad_w[idx]
                    if abs(w) > threshold:
                        if i == j:
                            quad_terms.append((abs(w), f"{w:.3f}*{var_names[i]}²"))
                        else:
                            quad_terms.append((abs(w), f"{w:.3f}*{var_names[i]}*{var_names[j]}"))
                    idx += 1
            
            # Sort by importance
            quad_terms.sort(reverse=True)
            terms.extend([t[1] for t in quad_terms[:5]])  # Top 5 quadratic terms
        
        if abs(bias) > threshold:
            terms.append(f"{bias:.3f}")
        
        return " + ".join(terms[:10]) if terms else "0"  # Limit to top 10 terms


def train_adaptive_symbolic(X_train, y_train, X_valid, y_valid, 
                            params: Dict[str, Any], verbose: bool = True):
    """Train symbolic regressor with adaptive hyperparameters."""
    
    model = AdaptiveSymbolicRegressor(
        n_features=N_DESCRIPTORS,
        max_degree=params.get('max_depth', 2),
        include_interactions=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32)
    
    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = params.get('patience', 30)
    l1_reg = params.get('l1_reg', 0.005)
    
    max_epochs = 500
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train_t)
        mse_loss = torch.mean((pred - y_train_t) ** 2)
        
        # L1 regularization for sparsity
        l1_loss = l1_reg * sum(p.abs().sum() for p in model.parameters())
        loss = mse_loss + l1_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid_t)
            valid_mse = torch.mean((valid_pred - y_valid_t) ** 2).item()
        
        if valid_mse < best_valid_loss - 0.0001:
            best_valid_loss = valid_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 100 == 0:
            train_r2 = 1 - mse_loss.item() / y_train_t.var().item()
            valid_r2 = 1 - valid_mse / y_valid_t.var().item()
            print(f"      Epoch {epoch:4d} | Train R²: {train_r2:.4f} | Valid R²: {valid_r2:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"      → Early stop at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final metrics
    model.eval()
    with torch.no_grad():
        valid_pred = model(X_valid_t).numpy()
    
    metrics = {
        'valid_rmse': float(np.sqrt(mean_squared_error(y_valid, valid_pred))),
        'valid_r2': float(r2_score(y_valid, valid_pred))
    }
    
    return model, metrics


def train_deepchem_symbolic(X_train, y_train, X_valid, y_valid, 
                            params: Dict[str, Any], verbose: bool = True):
    """Train DeepChem symbolic regressor with adaptive hyperparameters."""
    from models.symbolic_regressor import SymbolicRegressorModel
    
    train_ds = NumpyDataset(X=X_train, y=y_train.reshape(-1, 1))
    valid_ds = NumpyDataset(X=X_valid, y=y_valid.reshape(-1, 1))
    
    model = SymbolicRegressorModel(
        n_features=N_DESCRIPTORS,
        max_depth=params.get('max_depth', 2),
        n_candidates=params.get('n_candidates', 3),
        complexity_weight=params['complexity_weight'],
        learning_rate=params['learning_rate'],
        batch_size=params.get('batch_size', 32)
    )
    
    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = params.get('patience', 30)
    
    for epoch in range(200):
        model.fit(train_ds, nb_epoch=1, deterministic=False)
        
        valid_pred = model.predict(valid_ds).squeeze()
        if np.isnan(valid_pred).any():
            valid_pred = np.nan_to_num(valid_pred, nan=y_train.mean())
        
        valid_mse = mean_squared_error(y_valid, valid_pred)
        
        if valid_mse < best_valid_loss - 0.0001:
            best_valid_loss = valid_mse
            best_state = {k: v.clone() for k, v in model.model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 50 == 0:
            valid_r2 = r2_score(y_valid, valid_pred)
            print(f"      [DC] Epoch {epoch:3d} | Valid R²: {valid_r2:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"      [DC] → Early stop at epoch {epoch}")
            break
    
    if best_state:
        model.model.load_state_dict(best_state)
    
    valid_pred = model.predict(valid_ds).squeeze()
    valid_pred = np.nan_to_num(valid_pred, nan=y_train.mean())
    
    metrics = {
        'valid_rmse': float(np.sqrt(mean_squared_error(y_valid, valid_pred))),
        'valid_r2': float(r2_score(y_valid, valid_pred))
    }
    
    return model, metrics


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    """Run adaptive evaluation."""
    print("="*70)
    print("ADAPTIVE DESCRIPTOR-BASED SYMBOLIC REGRESSION EVALUATION")
    print("="*70)
    print(f"Date: {datetime.now()}")
    print(f"Descriptors ({N_DESCRIPTORS}): {DESCRIPTOR_NAMES}")
    print("\n⚡ Hyperparameters are ADAPTIVE per dataset (not fixed!)")
    
    all_results = {}
    
    for ds_name in ['delaney', 'lipo']:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name.upper()}")
        print(f"{'#'*70}")
        
        try:
            # Load and analyze dataset
            data = load_dataset(ds_name)
            X_train, y_train = data['X_train'], data['y_train']
            X_valid, y_valid = data['X_valid'], data['y_valid']
            X_test, y_test = data['X_test'], data['y_test']
            characteristics = data['characteristics']
            
            # ========== BASELINES ==========
            print("\n  BASELINES:")
            
            # Linear with adaptive regularization (RidgeCV)
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
            lr = RidgeCV(alphas=alphas, cv=5)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            lr_r2 = r2_score(y_test, lr_pred)
            print(f"    Linear (RidgeCV, α={lr.alpha_:.2f}): RMSE={lr_rmse:.4f}, R²={lr_r2:.4f}")
            
            # Random Forest with adaptive params
            n_est = 200 if characteristics['n_train'] > 1000 else 100
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)
            print(f"    Random Forest (n={n_est}):    RMSE={rf_rmse:.4f}, R²={rf_r2:.4f}")
            
            # ========== ADAPTIVE SYMBOLIC REGRESSION ==========
            print("\n  ADAPTIVE SYMBOLIC REGRESSION:")
            
            # Initialize tuner
            tuner = AdaptiveHyperparameterTuner(characteristics)
            
            # Tune hyperparameters
            best_params = tuner.tune(
                train_adaptive_symbolic,
                X_train, y_train, X_valid, y_valid,
                max_trials=6
            )
            
            # Train final model with best params
            print("\n  Training final model with best hyperparameters...")
            final_model, _ = train_adaptive_symbolic(
                X_train, y_train, X_valid, y_valid, 
                best_params, verbose=True
            )
            
            # Evaluate on test
            with torch.no_grad():
                test_pred = final_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            
            sr_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            sr_r2 = r2_score(y_test, test_pred)
            formula = final_model.get_formula(DESCRIPTOR_NAMES)
            
            print(f"\n  Test Results:")
            print(f"    RMSE: {sr_rmse:.4f}")
            print(f"    R²:   {sr_r2:.4f}")
            print(f"    Formula: {formula}")
            
            # ========== DEEPCHEM MODEL (with adapted params) ==========
            print("\n  DEEPCHEM SYMBOLIC (with adapted params):")
            dc_model, _ = train_deepchem_symbolic(
                X_train, y_train, X_valid, y_valid,
                best_params, verbose=True
            )
            
            test_ds = NumpyDataset(X=X_test, y=y_test.reshape(-1, 1))
            dc_pred = dc_model.predict(test_ds).squeeze()
            dc_pred = np.nan_to_num(dc_pred, nan=y_train.mean())
            
            dc_rmse = np.sqrt(mean_squared_error(y_test, dc_pred))
            dc_r2 = r2_score(y_test, dc_pred)
            dc_formula = dc_model.get_formula(var_names=DESCRIPTOR_NAMES)
            
            print(f"    Test RMSE: {dc_rmse:.4f}, R²: {dc_r2:.4f}")
            print(f"    Formula: {dc_formula[:80]}...")
            
            # ========== SUMMARY ==========
            print(f"\n  {'='*55}")
            print(f"  SUMMARY: {ds_name.upper()}")
            print(f"  {'='*55}")
            print(f"  Learned Hyperparameters:")
            print(f"    Learning rate: {best_params['learning_rate']}")
            print(f"    Complexity weight: {best_params['complexity_weight']}")
            print(f"    Max depth: {best_params['max_depth']}")
            print(f"    Patience: {best_params['patience']}")
            print(f"\n  {'Model':<25} {'RMSE':<10} {'R²':<10}")
            print(f"  {'-'*45}")
            print(f"  {'Linear (RidgeCV)':<25} {lr_rmse:<10.4f} {lr_r2:<10.4f}")
            print(f"  {'Random Forest':<25} {rf_rmse:<10.4f} {rf_r2:<10.4f}")
            print(f"  {'Adaptive Symbolic':<25} {sr_rmse:<10.4f} {sr_r2:<10.4f}")
            print(f"  {'DeepChem Symbolic':<25} {dc_rmse:<10.4f} {dc_r2:<10.4f}")
            
            all_results[ds_name] = {
                'characteristics': characteristics,
                'learned_hyperparameters': best_params,
                'tuning_history': tuner.search_history,
                'linear': {'rmse': float(lr_rmse), 'r2': float(lr_r2), 'alpha': float(lr.alpha_)},
                'random_forest': {'rmse': float(rf_rmse), 'r2': float(rf_r2)},
                'adaptive_symbolic': {
                    'rmse': float(sr_rmse),
                    'r2': float(sr_r2),
                    'formula': formula
                },
                'deepchem_symbolic': {
                    'rmse': float(dc_rmse),
                    'r2': float(dc_r2),
                    'formula': dc_formula
                }
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== CROSS-DATASET ANALYSIS ==========
    print("\n" + "="*70)
    print("CROSS-DATASET ANALYSIS: ADAPTIVE HYPERPARAMETERS")
    print("="*70)
    
    if len(all_results) == 2:
        print("\n  Learned Hyperparameters per Dataset:")
        print(f"  {'Parameter':<20} {'Delaney':<15} {'Lipo':<15}")
        print(f"  {'-'*50}")
        
        d_params = all_results['delaney']['learned_hyperparameters']
        l_params = all_results['lipo']['learned_hyperparameters']
        
        for key in ['learning_rate', 'complexity_weight', 'max_depth', 'patience']:
            print(f"  {key:<20} {d_params.get(key, 'N/A'):<15} {l_params.get(key, 'N/A'):<15}")
        
        print("\n  Why Different Hyperparameters?")
        print(f"  {'Characteristic':<25} {'Delaney':<15} {'Lipo':<15}")
        print(f"  {'-'*55}")
        
        d_char = all_results['delaney']['characteristics']
        l_char = all_results['lipo']['characteristics']
        
        print(f"  {'Training samples':<25} {d_char['n_train']:<15} {l_char['n_train']:<15}")
        print(f"  {'Samples/feature':<25} {d_char['samples_per_feature']:.1f!s:<15} {l_char['samples_per_feature']:.1f!s:<15}")
        print(f"  {'Linear baseline R²':<25} {d_char['linear_baseline_r2']:.3f!s:<15} {l_char['linear_baseline_r2']:.3f!s:<15}")
        print(f"  {'Problem difficulty':<25} {d_char['problem_difficulty']:<15} {l_char['problem_difficulty']:<15}")
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'adaptive_results')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'adaptive_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved: {filepath}")
    
    return all_results


if __name__ == "__main__":
    main()

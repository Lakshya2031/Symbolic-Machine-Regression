"""
Optimized Descriptor-Based Symbolic Regression Evaluation
==========================================================
Properly tuned evaluation with chemically meaningful RDKit descriptors.
"""

import sys
import os
import json
import warnings
from datetime import datetime
import numpy as np

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

from sklearn.linear_model import Ridge
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


def load_dataset(dataset_name: str):
    """Load MoleculeNet dataset with descriptors."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} ({N_DESCRIPTORS} descriptors)")
    print(f"{'='*60}")
    
    if dataset_name == 'delaney':
        from deepchem.molnet import load_delaney
        tasks, datasets, _ = load_delaney(featurizer='Raw', splitter='scaffold')
    else:
        from deepchem.molnet import load_lipo
        tasks, datasets, _ = load_lipo(featurizer='Raw', splitter='scaffold')
    
    train_ds, valid_ds, test_ds = datasets
    scaler = StandardScaler()
    
    print(f"  Featurizing {len(train_ds.ids)} train...")
    X_train = np.array([featurize_molecule(s) for s in train_ds.ids])
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    
    print(f"  Featurizing {len(valid_ds.ids)} valid...")
    X_valid = np.array([featurize_molecule(s) for s in valid_ds.ids])
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
    X_valid = scaler.transform(X_valid).astype(np.float32)
    
    print(f"  Featurizing {len(test_ds.ids)} test...")
    X_test = np.array([featurize_molecule(s) for s in test_ds.ids])
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    y_train = train_ds.y.squeeze().astype(np.float32)
    y_valid = valid_ds.y.squeeze().astype(np.float32)
    y_test = test_ds.y.squeeze().astype(np.float32)
    
    # Normalize targets
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_valid_norm = (y_valid - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    print(f"  ✓ Shapes: Train {X_train.shape}, Valid {X_valid.shape}, Test {X_test.shape}")
    print(f"  ✓ Y range: [{y_train.min():.2f}, {y_train.max():.2f}], mean={y_mean:.2f}")
    
    return (X_train, y_train, y_train_norm,
            X_valid, y_valid, y_valid_norm,
            X_test, y_test, y_test_norm,
            y_mean, y_std)


# =============================================================================
# SIMPLIFIED SYMBOLIC REGRESSION (Direct Training)
# =============================================================================

class SimpleSymbolicRegressor(nn.Module):
    """
    A simpler symbolic regression model for chemoinformatics.
    Uses polynomial combinations with learned weights.
    """
    
    def __init__(self, n_features: int, max_degree: int = 2):
        super().__init__()
        self.n_features = n_features
        self.max_degree = max_degree
        
        # Linear terms
        self.linear = nn.Linear(n_features, 1)
        
        # Quadratic terms (x_i * x_j for i <= j)
        n_quad = (n_features * (n_features + 1)) // 2
        self.quad_weight = nn.Parameter(torch.zeros(n_quad))
        
        # Feature importance for interpretability
        self.feature_importance = nn.Parameter(torch.ones(n_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear contribution
        out = self.linear(x).squeeze(-1)
        
        # Quadratic contribution
        quad_terms = []
        for i in range(self.n_features):
            for j in range(i, self.n_features):
                quad_terms.append(x[:, i] * x[:, j])
        
        if quad_terms:
            quad = torch.stack(quad_terms, dim=1)
            out = out + (quad * self.quad_weight).sum(dim=1)
        
        return out
    
    def get_formula(self, var_names):
        """Extract interpretable formula."""
        linear_weights = self.linear.weight.data.squeeze().cpu().numpy()
        bias = self.linear.bias.data.item()
        quad_weights = self.quad_weight.data.cpu().numpy()
        
        terms = []
        
        # Linear terms
        for i, (w, name) in enumerate(zip(linear_weights, var_names)):
            if abs(w) > 0.01:
                terms.append(f"{w:.3f}*{name}")
        
        # Quadratic terms
        idx = 0
        for i in range(self.n_features):
            for j in range(i, self.n_features):
                w = quad_weights[idx]
                if abs(w) > 0.01:
                    if i == j:
                        terms.append(f"{w:.3f}*{var_names[i]}²")
                    else:
                        terms.append(f"{w:.3f}*{var_names[i]}*{var_names[j]}")
                idx += 1
        
        if abs(bias) > 0.01:
            terms.append(f"{bias:.3f}")
        
        return " + ".join(terms) if terms else "0"


def train_simple_symbolic(X_train, y_train, X_valid, y_valid,
                          max_epochs=500, patience=50, lr=0.01, l1_reg=0.001):
    """Train simple symbolic regressor with regularization."""
    
    model = SimpleSymbolicRegressor(N_DESCRIPTORS, max_degree=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32)
    
    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward
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
        
        if epoch % 100 == 0:
            train_r2 = 1 - mse_loss.item() / y_train_t.var().item()
            valid_r2 = 1 - valid_mse / y_valid_t.var().item()
            print(f"    Epoch {epoch:4d} | Train R²: {train_r2:.4f} | Valid R²: {valid_r2:.4f}")
        
        if patience_counter >= patience:
            print(f"    → Early stop at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


def train_deepchem_symbolic(X_train, y_train, X_valid, y_valid):
    """Train using DeepChem SymbolicRegressorModel."""
    from models.symbolic_regressor import SymbolicRegressorModel
    
    train_ds = NumpyDataset(X=X_train, y=y_train.reshape(-1, 1))
    valid_ds = NumpyDataset(X=X_valid, y=y_valid.reshape(-1, 1))
    
    model = SymbolicRegressorModel(
        n_features=N_DESCRIPTORS,
        max_depth=2,  # Very simple
        n_candidates=3,
        complexity_weight=0.05,
        learning_rate=0.005,
        batch_size=64
    )
    
    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 30
    
    for epoch in range(150):
        model.fit(train_ds, nb_epoch=1, deterministic=False)
        
        train_pred = model.predict(train_ds).squeeze()
        valid_pred = model.predict(valid_ds).squeeze()
        
        train_r2 = r2_score(y_train, train_pred) if not np.isnan(train_pred).any() else -999
        valid_mse = mean_squared_error(y_valid, valid_pred) if not np.isnan(valid_pred).any() else 1e10
        valid_r2 = r2_score(y_valid, valid_pred) if not np.isnan(valid_pred).any() else -999
        
        if valid_mse < best_valid_loss - 0.0001:
            best_valid_loss = valid_mse
            best_state = {k: v.clone() for k, v in model.model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 30 == 0:
            print(f"    [DC] Epoch {epoch:3d} | Train R²: {train_r2:.4f} | Valid R²: {valid_r2:.4f}")
        
        if patience_counter >= patience:
            print(f"    [DC] → Early stop at epoch {epoch}")
            break
    
    if best_state:
        model.model.load_state_dict(best_state)
    
    return model


def main():
    """Main evaluation."""
    print("="*70)
    print("DESCRIPTOR-BASED SYMBOLIC REGRESSION EVALUATION")
    print("="*70)
    print(f"Date: {datetime.now()}")
    print(f"Descriptors ({N_DESCRIPTORS}): {DESCRIPTOR_NAMES}")
    
    all_results = {}
    
    for ds_name in ['delaney', 'lipo']:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name.upper()}")
        print(f"{'#'*70}")
        
        try:
            # Load
            (X_train, y_train, y_train_n,
             X_valid, y_valid, y_valid_n,
             X_test, y_test, y_test_n,
             y_mean, y_std) = load_dataset(ds_name)
            
            # ========== BASELINES ==========
            print("\n  BASELINES:")
            
            # Linear
            lr = Ridge(alpha=1.0)
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            lr_r2 = r2_score(y_test, lr_pred)
            print(f"    Linear:        RMSE={lr_rmse:.4f}, R²={lr_r2:.4f}")
            
            # RF
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)
            print(f"    Random Forest: RMSE={rf_rmse:.4f}, R²={rf_r2:.4f}")
            
            # ========== SIMPLE SYMBOLIC ==========
            print("\n  SIMPLE SYMBOLIC (Polynomial with L1):")
            simple_model = train_simple_symbolic(X_train, y_train, X_valid, y_valid)
            
            with torch.no_grad():
                simple_pred = simple_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            
            simple_rmse = np.sqrt(mean_squared_error(y_test, simple_pred))
            simple_r2 = r2_score(y_test, simple_pred)
            simple_formula = simple_model.get_formula(DESCRIPTOR_NAMES)
            
            print(f"    Test RMSE: {simple_rmse:.4f}, R²: {simple_r2:.4f}")
            print(f"    Formula: {simple_formula[:100]}...")
            
            # ========== DEEPCHEM SYMBOLIC ==========
            print("\n  DEEPCHEM SYMBOLIC REGRESSOR:")
            dc_model = train_deepchem_symbolic(X_train, y_train, X_valid, y_valid)
            
            test_ds = NumpyDataset(X=X_test, y=y_test.reshape(-1, 1))
            dc_pred = dc_model.predict(test_ds).squeeze()
            dc_pred = np.nan_to_num(dc_pred, nan=y_mean)
            
            dc_rmse = np.sqrt(mean_squared_error(y_test, dc_pred))
            dc_r2 = r2_score(y_test, dc_pred)
            dc_formula = dc_model.get_formula(var_names=DESCRIPTOR_NAMES)
            dc_complexity = dc_model.get_complexity()
            
            print(f"    Test RMSE: {dc_rmse:.4f}, R²: {dc_r2:.4f}")
            print(f"    Complexity: {dc_complexity:.2f}")
            print(f"    Formula: {dc_formula[:100]}...")
            
            # ========== RESULTS ==========
            print(f"\n  {'='*55}")
            print(f"  SUMMARY: {ds_name.upper()}")
            print(f"  {'='*55}")
            print(f"    {'Model':<25} {'RMSE':<10} {'R²':<10}")
            print(f"    {'-'*45}")
            print(f"    {'Linear Regression':<25} {lr_rmse:<10.4f} {lr_r2:<10.4f}")
            print(f"    {'Random Forest':<25} {rf_rmse:<10.4f} {rf_r2:<10.4f}")
            print(f"    {'Simple Symbolic':<25} {simple_rmse:<10.4f} {simple_r2:<10.4f}")
            print(f"    {'DeepChem Symbolic':<25} {dc_rmse:<10.4f} {dc_r2:<10.4f}")
            
            all_results[ds_name] = {
                'linear': {'rmse': float(lr_rmse), 'r2': float(lr_r2)},
                'random_forest': {'rmse': float(rf_rmse), 'r2': float(rf_r2)},
                'simple_symbolic': {
                    'rmse': float(simple_rmse), 
                    'r2': float(simple_r2),
                    'formula': simple_formula
                },
                'deepchem_symbolic': {
                    'rmse': float(dc_rmse),
                    'r2': float(dc_r2),
                    'complexity': float(dc_complexity),
                    'formula': dc_formula
                }
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== FINAL ANALYSIS ==========
    print("\n" + "="*70)
    print("ANALYSIS: DESCRIPTORS vs ECFP FOR SYMBOLIC REGRESSION")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ WHY DESCRIPTORS ENABLE SYMBOLIC REGRESSION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ECFP (1024 bits)              → Descriptors (15 features)           │
│ ─────────────────             ─────────────────────────             │
│ • Sparse binary (0/1)         • Dense continuous                    │
│ • Non-interpretable bits      • Chemical meaning (LogP, TPSA)       │
│ • 1024D search space          • 15D search space (tractable!)       │
│ • DP cache misses             • DP cache hits (reusable)            │
│                                                                     │
│ RESULT: Symbolic regression can discover meaningful relationships   │
│ like: Solubility ≈ a·LogP + b·TPSA + c·MolWt                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

PERFORMANCE COMPARISON:
──────────────────────
| Feature Type | Delaney R² | Lipo R² | Interpretable? | DP Benefits? |
|--------------|------------|---------|----------------|--------------|
| ECFP (1024)  | 0.47       | 0.08    | ✗ No          | ✗ No         |""")
    
    if 'delaney' in all_results:
        d_r2 = all_results['delaney']['simple_symbolic']['r2']
    else:
        d_r2 = "N/A"
    
    if 'lipo' in all_results:
        l_r2 = all_results['lipo']['simple_symbolic']['r2']
    else:
        l_r2 = "N/A"
    
    if isinstance(d_r2, float) and isinstance(l_r2, float):
        print(f"| Descriptors  | {d_r2:.2f}       | {l_r2:.2f}    | ✓ Yes          | ✓ Yes        |")
    
    print("""
──────────────────────────────────────────────────────────────────────

DISCOVERED FORMULAS (Chemically Interpretable):
───────────────────────────────────────────────""")
    
    for ds_name, results in all_results.items():
        print(f"\n{ds_name.upper()}:")
        if 'simple_symbolic' in results:
            print(f"  {results['simple_symbolic']['formula'][:80]}...")
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'descriptor_results')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved: {filepath}")
    
    return all_results


if __name__ == "__main__":
    main()

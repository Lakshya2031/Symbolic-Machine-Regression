"""
Main Script for Symbolic Regression
===================================
Run this script to discover equations from your data.
The model learns the structure automatically - no manual fitting!

Usage:
    python main.py
    
Then edit the DATA section below with your own data.
"""

import torch
import numpy as np
from model import SymbolicRegressor, MultiTermRegressor
from trainer import SymbolicRegressionTrainer
from simplify import print_expression_report

# =============================================================================
# STEP 1: ENTER YOUR DATA HERE
# =============================================================================

# Format: Each row is one sample [x1, x2, ...] 
# Add more columns if you have more input variables

X_DATA = [
    [1.2, -0.5],
    [-2.0, 1.5],
    [0.7, -2.1],
    [3.1, 0.8],
]

# Output values (one per row of X_DATA)
Y_DATA = [4.01, 15.38, -7.93, 29.25]

# Names for your variables (optional, for nicer output)
VARIABLE_NAMES = ["x1", "x2"]

# =============================================================================
# STEP 2: TRAINING SETTINGS (adjust if needed)
# =============================================================================

MAX_DEPTH = 4          # Maximum expression tree depth (2-4 recommended)
N_CANDIDATES = 8       # Number of candidate expressions to try
N_EPOCHS = 4000        # Training iterations
LEARNING_RATE = 0.015  # Learning rate
COMPLEXITY_WEIGHT = 0.0001  # Penalty for complex expressions (higher = simpler)

# =============================================================================
# STEP 3: RUN THE MODEL (no changes needed below)
# =============================================================================

def augment_data(x, y, n_augment=500, noise=0.01):
    """
    Generate additional training samples by interpolation.
    Helps the model learn better with limited data.
    """
    n_features = x.shape[1]
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    
    # Expand range slightly
    margin = (x_max - x_min) * 0.2
    x_min = x_min - margin
    x_max = x_max + margin
    
    # Generate random points in the range
    x_aug = torch.rand(n_augment, n_features) * (x_max - x_min) + x_min
    
    # We don't know y for augmented data, so we'll use original data only
    # But add small noise to original data for regularization
    x_noisy = x + torch.randn_like(x) * noise * (x_max - x_min)
    y_noisy = y + torch.randn_like(y) * noise * y.std()
    
    return torch.cat([x, x_noisy]), torch.cat([y, y_noisy])


def main():
    print("=" * 70)
    print("SYMBOLIC REGRESSION - AUTOMATIC EQUATION DISCOVERY")
    print("=" * 70)
    
    # Convert data to tensors
    x = torch.tensor(X_DATA, dtype=torch.float32)
    y = torch.tensor(Y_DATA, dtype=torch.float32)
    
    n_samples, n_features = x.shape
    
    print(f"\nInput Data:")
    print(f"  - Number of samples: {n_samples}")
    print(f"  - Number of features: {n_features}")
    print(f"  - Variables: {VARIABLE_NAMES[:n_features]}")
    print(f"\nData Preview:")
    for i in range(min(5, n_samples)):
        vars_str = ", ".join([f"{VARIABLE_NAMES[j]}={x[i,j]:.2f}" for j in range(n_features)])
        print(f"  [{vars_str}] -> y = {y[i]:.2f}")
    if n_samples > 5:
        print(f"  ... and {n_samples - 5} more samples")
    
    # Augment data if we have few samples
    if n_samples < 50:
        print(f"\nAugmenting data (original: {n_samples} samples)...")
        x_train, y_train = augment_data(x, y, n_augment=0, noise=0.02)
        # For very small datasets, just use original with noise
        x_train = torch.cat([x] * 50)  # Repeat data
        y_train = torch.cat([y] * 50)
        print(f"  Training with {len(x_train)} samples")
    else:
        x_train, y_train = x, y
    
    # Create model
    print(f"\nCreating model...")
    print(f"  - Max depth: {MAX_DEPTH}")
    print(f"  - Candidates: {N_CANDIDATES}")
    
    model = SymbolicRegressor(
        n_features=n_features,
        max_depth=MAX_DEPTH,
        n_candidates=N_CANDIDATES,
        complexity_weight=COMPLEXITY_WEIGHT
    )
    
    # Create trainer
    trainer = SymbolicRegressionTrainer(
        model=model,
        optimizer="adam",
        learning_rate=LEARNING_RATE,
        complexity_weight=COMPLEXITY_WEIGHT,
        complexity_schedule="warmup"
    )
    
    # Train
    print(f"\nTraining model ({N_EPOCHS} epochs)...")
    print("-" * 70)
    
    results = trainer.fit(
        x_train, y_train,
        n_epochs=N_EPOCHS,
        batch_size=min(32, len(x_train)),
        verbose=1,
        print_every=500,
        early_stopping_patience=500
    )
    
    # Print learned expression
    print("\n")
    print_expression_report(model, var_names=VARIABLE_NAMES[:n_features])
    
    # Verify on original data
    print("\n" + "=" * 70)
    print("VERIFICATION ON YOUR DATA")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        predictions = model(x)
    
    print(f"\n{'Input':<30} {'Predicted':<15} {'Actual':<15} {'Error':<10}")
    print("-" * 70)
    
    total_error = 0
    for i in range(n_samples):
        vars_str = ", ".join([f"{x[i,j]:.2f}" for j in range(n_features)])
        pred = predictions[i].item()
        actual = y[i].item()
        error = abs(pred - actual)
        total_error += error ** 2
        print(f"f({vars_str}){'':<{20-len(vars_str)}} {pred:<15.4f} {actual:<15.4f} {error:<10.4f}")
    
    rmse = np.sqrt(total_error / n_samples)
    print("-" * 70)
    print(f"Root Mean Square Error: {rmse:.6f}")
    
    # Final expression
    print("\n" + "=" * 70)
    print("DISCOVERED EQUATION")
    print("=" * 70)
    simplified = model.simplify(var_names=VARIABLE_NAMES[:n_features])
    print(f"\n  f({', '.join(VARIABLE_NAMES[:n_features])}) = {simplified}")
    print("\n" + "=" * 70)
    
    return model, results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, results = main()

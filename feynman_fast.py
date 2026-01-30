"""
Fast Feynman Benchmark
======================
Quick test on 3 Feynman equations with reduced training.
"""

import torch
import numpy as np
import time
from model import SymbolicRegressor
from trainer import SymbolicRegressionTrainer

torch.manual_seed(42)
np.random.seed(42)

# 3 Simple Feynman equations
EQUATIONS = {
    "I.6.2 (Kinetic Energy)": {
        "func": lambda x: 0.5 * x[:, 0] * x[:, 1]**2,
        "vars": ["m", "v"],
        "true": "0.5 * m * v²",
        "range": [(1, 5), (1, 5)]
    },
    "I.12.1 (Electric Field)": {
        "func": lambda x: x[:, 0] / (x[:, 1]**2),
        "vars": ["q", "r"],
        "true": "q / r²",
        "range": [(1, 5), (1, 5)]
    },
    "I.29.4 (Wave Number)": {
        "func": lambda x: x[:, 0] / x[:, 1],
        "vars": ["ω", "c"],
        "true": "ω / c",
        "range": [(1, 10), (1, 10)]
    },
}

def run_benchmark():
    print("=" * 60)
    print("FEYNMAN BENCHMARK (Fast Version)")
    print("=" * 60)
    
    results = []
    
    for name, eq in EQUATIONS.items():
        print(f"\n{'-' * 60}")
        print(f"Testing: {name}")
        print(f"True equation: {eq['true']}")
        print(f"{'-' * 60}")
        
        # Generate data
        n_features = len(eq['vars'])
        x = torch.zeros(300, n_features)
        for i in range(n_features):
            low, high = eq['range'][i]
            x[:, i] = torch.rand(300) * (high - low) + low
        y = eq['func'](x)
        
        # Create & train model
        model = SymbolicRegressor(
            n_features=n_features,
            max_depth=2,
            n_candidates=3,
            complexity_weight=0.001
        )
        
        trainer = SymbolicRegressionTrainer(
            model=model,
            learning_rate=0.03,
            complexity_weight=0.001
        )
        
        start = time.time()
        trainer.fit(x, y, n_epochs=800, batch_size=32, verbose=0, 
                   early_stopping_patience=200)
        train_time = time.time() - start
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(x)
            mse = torch.mean((y_pred - y)**2).item()
            ss_res = torch.sum((y - y_pred)**2)
            ss_tot = torch.sum((y - torch.mean(y))**2)
            r2 = (1 - ss_res / ss_tot).item()
        
        learned = model.simplify(eq['vars'])
        
        print(f"  Learned:  {learned}")
        print(f"  R²: {r2:.4f} | MSE: {mse:.6f} | Time: {train_time:.1f}s")
        
        quality = "✓ GOOD" if r2 > 0.95 else ("○ OK" if r2 > 0.8 else "✗ POOR")
        print(f"  Quality: {quality}")
        
        results.append({"name": name, "r2": r2, "mse": mse})
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    r2_scores = [r["r2"] for r in results]
    print(f"Mean R²: {np.mean(r2_scores):.4f}")
    print(f"Equations with R² > 0.95: {sum(1 for r in r2_scores if r > 0.95)}/3")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()

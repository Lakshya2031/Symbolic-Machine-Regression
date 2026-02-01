"""
Example: DeepChem Integration

Demonstrates how to use the Symbolic Regression model
with DeepChem's dataset and model infrastructure.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import deepchem as dc
    print(f"✓ DeepChem version: {dc.__version__}")
    DEEPCHEM_AVAILABLE = True
except ImportError:
    print("✗ DeepChem not installed. Install with: pip install deepchem")
    DEEPCHEM_AVAILABLE = False

if DEEPCHEM_AVAILABLE:
    from deepchem_integration import SymbolicRegressorModel, feynman_to_dataset, create_symbolic_dataset
    
    def example_1_basic_usage():
        """Basic usage with NumpyDataset."""
        print("\n" + "="*60)
        print("Example 1: Basic Usage with NumpyDataset")
        print("="*60)
        
        # Generate data: y = 2*x0 + x1^2
        np.random.seed(42)
        X = np.random.randn(500, 2).astype(np.float32)
        y = (2 * X[:, 0] + X[:, 1]**2).astype(np.float32)
        
        # Create DeepChem dataset
        dataset = dc.data.NumpyDataset(X=X, y=y)
        print(f"Dataset shape: X={dataset.X.shape}, y={dataset.y.shape}")
        
        # Create model
        model = SymbolicRegressorModel(
            n_features=2,
            max_depth=3,
            learning_rate=0.02,
            batch_size=64
        )
        
        # Train
        print("\nTraining...")
        loss = model.fit(dataset, nb_epoch=100)
        print(f"Final loss: {loss:.6f}")
        
        # Get formula
        formula = model.get_formula(var_names=['a', 'b'])
        print(f"\nTarget:     y = 2*a + b^2")
        print(f"Discovered: {formula}")
        
        # Evaluate
        predictions = model.predict(dataset)
        mse = np.mean((predictions.squeeze() - y)**2)
        print(f"MSE: {mse:.6f}")
    
    def example_2_feynman_equations():
        """Using Feynman benchmark equations."""
        print("\n" + "="*60)
        print("Example 2: Feynman Benchmark Equations")
        print("="*60)
        
        # Test on kinetic energy: E = 0.5 * m * v^2
        dataset, info = feynman_to_dataset('I.6.2', n_samples=1000, seed=42)
        
        print(f"\nEquation: {info['equation_id']} ({info['description']})")
        print(f"Target formula: {info['formula']}")
        print(f"Variables: {info['var_names']}")
        
        # Create and train model
        model = SymbolicRegressorModel(
            n_features=info['n_features'],
            max_depth=4,
            learning_rate=0.01
        )
        
        print("\nTraining...")
        loss = model.fit(dataset, nb_epoch=150)
        
        # Results
        formula = model.get_formula(var_names=info['var_names'])
        print(f"\nDiscovered: {formula}")
        print(f"Complexity: {model.get_complexity():.2f}")
        
        # Evaluate
        predictions = model.predict(dataset)
        r2 = 1 - np.sum((predictions.squeeze() - dataset.y.squeeze())**2) / np.sum((dataset.y.squeeze() - np.mean(dataset.y))**2)
        print(f"R² score: {r2:.4f}")
    
    def example_3_custom_function():
        """Using create_symbolic_dataset with custom function."""
        print("\n" + "="*60)
        print("Example 3: Custom Function Dataset")
        print("="*60)
        
        # Define a custom function
        def custom_func(X):
            return np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
        
        dataset = create_symbolic_dataset(
            func=custom_func,
            n_samples=1000,
            n_features=3,
            x_range=(-3.0, 3.0),
            noise_level=0.01,
            seed=123
        )
        
        print(f"Target: y = sin(x0) + x1 * x2")
        print(f"Dataset: {len(dataset)} samples, {dataset.X.shape[1]} features")
        
        model = SymbolicRegressorModel(
            n_features=3,
            max_depth=4,
            n_candidates=7
        )
        
        print("\nTraining...")
        loss = model.fit(dataset, nb_epoch=200)
        
        formula = model.get_formula(var_names=['x', 'y', 'z'])
        print(f"\nDiscovered: {formula}")
        
        # Show all candidates
        print("\nCandidate expressions:")
        info = model.get_candidate_info()
        for cand in sorted(info['candidates'], key=lambda x: -x['weight'])[:3]:
            print(f"  Weight {cand['weight']:.3f}: {cand['expression']}")
    
    def main():
        print("DeepChem Integration Examples")
        print("=" * 60)
        
        example_1_basic_usage()
        example_2_feynman_equations()
        example_3_custom_function()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
    
    if __name__ == "__main__":
        main()

else:
    print("\nPlease install DeepChem to run these examples:")
    print("  pip install deepchem")

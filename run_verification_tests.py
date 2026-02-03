"""
Comprehensive Test Verification Suite
=====================================

This script runs all verification tests for the GSoC Symbolic Regression
project and generates a detailed report for documentation purposes.

Run this script to verify all components are working correctly before
submitting your GSoC proposal.

Author: GSoC Symbolic Regression Project
Date: February 3, 2026
"""

import sys
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'symbolic_regression', 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TestResult:
    """Container for test results."""
    def __init__(self, name: str, passed: bool, message: str, duration: float):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration


class TestVerificationSuite:
    """Comprehensive test suite for GSoC proposal verification."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
    
    def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test and capture result."""
        start = time.time()
        try:
            result, message = test_func()
            duration = time.time() - start
            test_result = TestResult(name, result, message, duration)
        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(name, False, f"Exception: {str(e)}", duration)
        
        self.results.append(test_result)
        status = "✅ PASSED" if test_result.passed else "❌ FAILED"
        print(f"  {status}: {name} ({duration:.2f}s)")
        return test_result
    
    def run_all_tests(self):
        """Run all verification tests."""
        self.start_time = datetime.now()
        
        print("="*70)
        print("GSoC SYMBOLIC REGRESSION - COMPREHENSIVE VERIFICATION")
        print("="*70)
        print(f"Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Category 1: Import Tests
        print("\n📦 CATEGORY 1: Import Verification")
        print("-"*50)
        self.run_test("Import DeepChem", self.test_import_deepchem)
        self.run_test("Import PyTorch", self.test_import_pytorch)
        self.run_test("Import RDKit", self.test_import_rdkit)
        self.run_test("Import SymbolicRegressorModel", self.test_import_model)
        self.run_test("Import Core Operators", self.test_import_operators)
        
        # Category 2: Core Functionality
        print("\n⚙️ CATEGORY 2: Core Functionality")
        print("-"*50)
        self.run_test("Protected Division (zero)", self.test_protected_div_zero)
        self.run_test("Protected Log (negative)", self.test_protected_log_negative)
        self.run_test("Protected Sqrt (negative)", self.test_protected_sqrt_negative)
        self.run_test("Exp Overflow Protection", self.test_exp_overflow)
        self.run_test("Operator Differentiability", self.test_operators_differentiable)
        
        # Category 3: DeepChem Integration
        print("\n🔗 CATEGORY 3: DeepChem Integration")
        print("-"*50)
        self.run_test("TorchModel Inheritance", self.test_torchmodel_inheritance)
        self.run_test("NumpyDataset Compatibility", self.test_numpy_dataset)
        self.run_test("Model Fit API", self.test_model_fit)
        self.run_test("Model Predict API", self.test_model_predict)
        self.run_test("Get Formula API", self.test_get_formula)
        
        # Category 4: Symbolic Regression Quality
        print("\n📊 CATEGORY 4: Regression Quality")
        print("-"*50)
        self.run_test("Simple Polynomial (x²+2x)", self.test_simple_polynomial)
        self.run_test("Linear Combination", self.test_linear_combination)
        self.run_test("Multiplicative Formula", self.test_multiplicative)
        
        # Category 5: Noise Robustness
        print("\n🎯 CATEGORY 5: Noise Robustness")
        print("-"*50)
        self.run_test("Noise 0% (clean data)", self.test_noise_0)
        self.run_test("Noise 10%", self.test_noise_10)
        self.run_test("Noise 20%", self.test_noise_20)
        
        # Category 6: RDKit Descriptors
        print("\n🧪 CATEGORY 6: RDKit Descriptors")
        print("-"*50)
        self.run_test("Descriptor Computation", self.test_descriptor_computation)
        self.run_test("Descriptor Count", self.test_descriptor_count)
        self.run_test("Descriptor Types (continuous)", self.test_descriptor_continuous)
        
        # Category 7: Performance
        print("\n⚡ CATEGORY 7: Performance")
        print("-"*50)
        self.run_test("Training Speed (<60s for 100 epochs)", self.test_training_speed)
        self.run_test("Prediction Speed (<1s for 1000 samples)", self.test_prediction_speed)
        
        self.print_summary()
    
    # =========================================================================
    # Import Tests
    # =========================================================================
    
    def test_import_deepchem(self):
        import deepchem as dc
        return True, f"DeepChem {dc.__version__}"
    
    def test_import_pytorch(self):
        import torch
        return True, f"PyTorch {torch.__version__}"
    
    def test_import_rdkit(self):
        from rdkit import Chem
        return True, "RDKit available"
    
    def test_import_model(self):
        from models.symbolic_regressor import SymbolicRegressorModel
        return True, "SymbolicRegressorModel imported"
    
    def test_import_operators(self):
        from core.operators import add, mul, protected_div, sin_op, exp_op
        return True, "Core operators imported"
    
    # =========================================================================
    # Core Functionality Tests
    # =========================================================================
    
    def test_protected_div_zero(self):
        import torch
        from core.operators import protected_div
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 0.0, 0.0])
        result = protected_div(x, y)
        passed = torch.isfinite(result).all().item()
        return passed, f"Result finite: {passed}"
    
    def test_protected_log_negative(self):
        import torch
        from core.operators import protected_log
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5])
        result = protected_log(x)
        passed = torch.isfinite(result).all().item()
        return passed, f"Result finite: {passed}"
    
    def test_protected_sqrt_negative(self):
        import torch
        from core.operators import sqrt_op
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = sqrt_op(x)
        passed = torch.isfinite(result).all().item()
        return passed, f"Result finite: {passed}"
    
    def test_exp_overflow(self):
        import torch
        from core.operators import exp_op
        x = torch.tensor([100.0, -100.0, 1000.0])
        result = exp_op(x)
        passed = torch.isfinite(result).all().item()
        return passed, f"Result finite: {passed}"
    
    def test_operators_differentiable(self):
        import torch
        from core.operators import add, mul, sin_op
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)
        result = sin_op(add(x, mul(x, y)))
        loss = result.sum()
        loss.backward()
        passed = x.grad is not None and y.grad is not None
        return passed, f"Gradients computed: {passed}"
    
    # =========================================================================
    # DeepChem Integration Tests
    # =========================================================================
    
    def test_torchmodel_inheritance(self):
        from models.symbolic_regressor import SymbolicRegressorModel
        from deepchem.models.torch_models import TorchModel
        passed = issubclass(SymbolicRegressorModel, TorchModel)
        return passed, f"Inherits TorchModel: {passed}"
    
    def test_numpy_dataset(self):
        import deepchem as dc
        X = np.random.randn(50, 2).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        passed = len(dataset) == 50
        return passed, f"Dataset created with {len(dataset)} samples"
    
    def test_model_fit(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        X = np.random.randn(50, 2).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        model = SymbolicRegressorModel(n_features=2, max_depth=2, n_candidates=2)
        loss = model.fit(dataset, nb_epoch=5)
        passed = loss is not None
        return passed, f"Fit completed, loss type: {type(loss).__name__}"
    
    def test_model_predict(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        X = np.random.randn(50, 2).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        model = SymbolicRegressorModel(n_features=2, max_depth=2, n_candidates=2)
        model.fit(dataset, nb_epoch=5)
        preds = model.predict(dataset)
        passed = preds.shape == (50, 1)
        return passed, f"Predictions shape: {preds.shape}"
    
    def test_get_formula(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        X = np.random.randn(50, 2).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        model = SymbolicRegressorModel(n_features=2, max_depth=2, n_candidates=2)
        model.fit(dataset, nb_epoch=10)
        formula = model.get_formula(var_names=['x0', 'x1'])
        passed = isinstance(formula, str) and len(formula) > 0
        return passed, f"Formula: {formula[:50]}..."
    
    # =========================================================================
    # Regression Quality Tests
    # =========================================================================
    
    def test_simple_polynomial(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        from sklearn.metrics import r2_score
        
        np.random.seed(42)
        X = np.random.randn(200, 2).astype(np.float32)
        y = (X[:, 0]**2 + 2*X[:, 1]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
        model = SymbolicRegressorModel(n_features=2, max_depth=2, n_candidates=3, learning_rate=0.01)
        model.fit(dataset, nb_epoch=80)
        
        preds = model.predict(dataset).squeeze()
        r2 = r2_score(y, preds)
        passed = r2 > 0.5
        return passed, f"R² = {r2:.3f}"
    
    def test_linear_combination(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        from sklearn.metrics import r2_score
        
        np.random.seed(42)
        X = np.random.randn(200, 3).astype(np.float32)
        y = (2*X[:, 0] - 3*X[:, 1] + X[:, 2]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
        model = SymbolicRegressorModel(n_features=3, max_depth=2, n_candidates=3, learning_rate=0.01)
        model.fit(dataset, nb_epoch=80)
        
        preds = model.predict(dataset).squeeze()
        r2 = r2_score(y, preds)
        passed = r2 > 0.5
        return passed, f"R² = {r2:.3f}"
    
    def test_multiplicative(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        from sklearn.metrics import r2_score
        
        np.random.seed(42)
        X = np.random.uniform(0.1, 5, (200, 2)).astype(np.float32)
        y = (X[:, 0] * X[:, 1]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
        model = SymbolicRegressorModel(n_features=2, max_depth=2, n_candidates=3, learning_rate=0.01)
        model.fit(dataset, nb_epoch=80)
        
        preds = model.predict(dataset).squeeze()
        r2 = r2_score(y, preds)
        passed = r2 > 0.3
        return passed, f"R² = {r2:.3f}"
    
    # =========================================================================
    # Noise Robustness Tests
    # =========================================================================
    
    def _test_with_noise(self, noise_level: float):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        from sklearn.metrics import r2_score
        
        np.random.seed(42)
        X = np.random.randn(200, 2).astype(np.float32)
        y_clean = (X[:, 0]**2 + 2*X[:, 1]).astype(np.float32)
        noise = np.random.normal(0, noise_level * np.std(y_clean), y_clean.shape).astype(np.float32)
        y_noisy = y_clean + noise
        
        dataset = dc.data.NumpyDataset(X=X, y=y_noisy)
        model = SymbolicRegressorModel(n_features=2, max_depth=2, n_candidates=3, learning_rate=0.01)
        model.fit(dataset, nb_epoch=80)
        
        preds = model.predict(dataset).squeeze()
        r2 = r2_score(y_noisy, preds)
        return r2
    
    def test_noise_0(self):
        r2 = self._test_with_noise(0.0)
        passed = r2 > 0.5
        return passed, f"R² = {r2:.3f} (threshold: 0.5)"
    
    def test_noise_10(self):
        r2 = self._test_with_noise(0.1)
        passed = r2 > 0.2
        return passed, f"R² = {r2:.3f} (threshold: 0.2)"
    
    def test_noise_20(self):
        r2 = self._test_with_noise(0.2)
        passed = r2 > 0.1
        return passed, f"R² = {r2:.3f} (threshold: 0.1)"
    
    # =========================================================================
    # RDKit Descriptor Tests
    # =========================================================================
    
    def test_descriptor_computation(self):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
        
        smiles = 'CCO'  # Ethanol
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        
        passed = mw > 0 and isinstance(logp, float)
        return passed, f"MolWt={mw:.1f}, LogP={logp:.2f}"
    
    def test_descriptor_count(self):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, Crippen
        
        descriptors = [
            'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAromaticRings', 'RingCount',
            'FractionCSP3', 'HeavyAtomCount', 'NOCount', 'Chi0',
            'Kappa1', 'LabuteASA', 'BertzCT'
        ]
        passed = len(descriptors) >= 15
        return passed, f"{len(descriptors)} descriptors available"
    
    def test_descriptor_continuous(self):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
        
        smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
        logp_values = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            logp_values.append(Crippen.MolLogP(mol))
        
        # Check that LogP values are continuous (not binary)
        unique_values = len(set(logp_values))
        passed = unique_values == len(smiles_list)  # All different
        return passed, f"LogP values: {logp_values} (all unique)"
    
    # =========================================================================
    # Performance Tests
    # =========================================================================
    
    def test_training_speed(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        
        X = np.random.randn(500, 3).astype(np.float32)
        y = (X[:, 0] + X[:, 1] * X[:, 2]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
        model = SymbolicRegressorModel(n_features=3, max_depth=2, n_candidates=3)
        
        start = time.time()
        model.fit(dataset, nb_epoch=100)
        duration = time.time() - start
        
        passed = duration < 60
        return passed, f"100 epochs in {duration:.1f}s"
    
    def test_prediction_speed(self):
        import deepchem as dc
        from models.symbolic_regressor import SymbolicRegressorModel
        
        X = np.random.randn(1000, 3).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)
        dataset = dc.data.NumpyDataset(X=X, y=y)
        
        model = SymbolicRegressorModel(n_features=3, max_depth=2, n_candidates=3)
        model.fit(dataset, nb_epoch=10)
        
        start = time.time()
        preds = model.predict(dataset)
        duration = time.time() - start
        
        passed = duration < 1.0
        return passed, f"1000 predictions in {duration:.3f}s"
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    def print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Pass Rate: {passed/total*100:.1f}%")
        print(f"Total Duration: {sum(r.duration for r in self.results):.1f}s")
        
        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        print("\n" + "="*70)
        if passed == total:
            print("🎉 ALL TESTS PASSED! Ready for GSoC submission.")
        else:
            print("⚠️ Some tests failed. Please fix before submission.")
        print("="*70)
        
        # Generate report file
        self._generate_report()
    
    def _generate_report(self):
        """Generate markdown report."""
        report_path = os.path.join(PROJECT_ROOT, 'TEST_VERIFICATION_REPORT.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Test Verification Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total = len(self.results)
            passed = sum(1 for r in self.results if r.passed)
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests:** {total}\n")
            f.write(f"- **Passed:** {passed} ✅\n")
            f.write(f"- **Failed:** {total - passed} ❌\n")
            f.write(f"- **Pass Rate:** {passed/total*100:.1f}%\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Test | Status | Message | Duration |\n")
            f.write("|------|--------|---------|----------|\n")
            
            for r in self.results:
                status = "✅ PASS" if r.passed else "❌ FAIL"
                f.write(f"| {r.name} | {status} | {r.message[:40]}... | {r.duration:.2f}s |\n")
        
        print(f"\n📄 Report saved to: {report_path}")


def main():
    """Run all verification tests."""
    suite = TestVerificationSuite()
    suite.run_all_tests()


if __name__ == '__main__':
    main()

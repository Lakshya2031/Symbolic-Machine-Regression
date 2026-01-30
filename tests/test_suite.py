"""
Comprehensive Test Suite
------------------------
Unit tests and integration tests for all modules in the symbolic
regression framework.

Test Categories:
    1. Unit Tests: Individual component testing
    2. Integration Tests: End-to-end workflow testing
    3. Regression Tests: Consistency and determinism
    4. Performance Tests: Speed and memory benchmarks
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestBaselineOperators(unittest.TestCase):
    """Tests for baseline operator implementations."""
    
    def test_binary_operators_shapes(self):
        from pysr_baseline.operators import add, sub, mul, protected_div
        x = torch.randn(10)
        y = torch.randn(10)
        
        self.assertEqual(add(x, y).shape, x.shape)
        self.assertEqual(sub(x, y).shape, x.shape)
        self.assertEqual(mul(x, y).shape, x.shape)
        self.assertEqual(protected_div(x, y).shape, x.shape)
    
    def test_protected_division_stability(self):
        from pysr_baseline.operators import protected_div
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 0.0, 0.0])
        
        result = protected_div(x, y)
        self.assertTrue(torch.isfinite(result).all())
    
    def test_unary_operators_shapes(self):
        from pysr_baseline.operators import sin_op, cos_op, exp_op, protected_log, sqrt_op
        x = torch.randn(10)
        
        self.assertEqual(sin_op(x).shape, x.shape)
        self.assertEqual(cos_op(x).shape, x.shape)
        self.assertEqual(exp_op(x).shape, x.shape)
        self.assertEqual(protected_log(x).shape, x.shape)
        self.assertEqual(sqrt_op(x).shape, x.shape)
    
    def test_protected_log_stability(self):
        from pysr_baseline.operators import protected_log
        x = torch.tensor([-1.0, 0.0, 1.0])
        
        result = protected_log(x)
        self.assertTrue(torch.isfinite(result).all())
    
    def test_exp_clamping(self):
        from pysr_baseline.operators import exp_op
        x = torch.tensor([100.0, -100.0])
        
        result = exp_op(x)
        self.assertTrue(torch.isfinite(result).all())


class TestBaselineNodes(unittest.TestCase):
    """Tests for baseline node implementations."""
    
    def test_variable_node(self):
        from pysr_baseline.nodes import VariableNode
        node = VariableNode(var_index=0)
        x = torch.randn(10, 3)
        
        output = node(x)
        self.assertEqual(output.shape, (10,))
        self.assertTrue(torch.allclose(output, x[:, 0]))
    
    def test_constant_node(self):
        from pysr_baseline.nodes import ConstantNode
        node = ConstantNode(init_value=5.0)
        x = torch.randn(10, 3)
        
        output = node(x)
        self.assertEqual(output.shape, (10,))
        self.assertTrue(torch.allclose(output, torch.full((10,), 5.0)))
    
    def test_constant_node_learnable(self):
        from pysr_baseline.nodes import ConstantNode
        node = ConstantNode(init_value=1.0)
        
        self.assertTrue(node.value.requires_grad)
    
    def test_weighted_input_node(self):
        from pysr_baseline.nodes import WeightedInputNode
        node = WeightedInputNode(n_features=3)
        x = torch.randn(10, 3)
        
        output = node(x)
        self.assertEqual(output.shape, (10,))
    
    def test_unary_op_node(self):
        from pysr_baseline.nodes import UnaryOpNode, WeightedInputNode
        child = WeightedInputNode(n_features=3)
        node = UnaryOpNode(child)
        x = torch.randn(10, 3)
        
        output = node(x)
        self.assertEqual(output.shape, (10,))
    
    def test_binary_op_node(self):
        from pysr_baseline.nodes import BinaryOpNode, WeightedInputNode
        left = WeightedInputNode(n_features=3)
        right = WeightedInputNode(n_features=3)
        node = BinaryOpNode(left, right)
        x = torch.randn(10, 3)
        
        output = node(x)
        self.assertEqual(output.shape, (10,))


class TestBaselineModel(unittest.TestCase):
    """Tests for baseline model implementation."""
    
    def test_model_creation(self):
        from pysr_baseline.model import SymbolicRegressor
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        
        self.assertEqual(len(model.candidates), 3)
    
    def test_forward_pass(self):
        from pysr_baseline.model import SymbolicRegressor
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        x = torch.randn(10, 3)
        
        output = model(x)
        self.assertEqual(output.shape, (10,))
    
    def test_gradient_flow(self):
        from pysr_baseline.model import SymbolicRegressor
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        x = torch.randn(10, 3)
        y = torch.randn(10)
        
        output = model(x)
        loss = torch.mean((output - y) ** 2)
        loss.backward()
        
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad)
    
    def test_complexity_computation(self):
        from pysr_baseline.model import SymbolicRegressor
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        
        complexity = model.get_complexity()
        self.assertGreater(complexity.item(), 0)
    
    def test_expression_generation(self):
        from pysr_baseline.model import SymbolicRegressor
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        
        expr = model.to_expression()
        self.assertIsInstance(expr, str)
        self.assertGreater(len(expr), 0)
    
    def test_simplify(self):
        from pysr_baseline.model import SymbolicRegressor
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        
        simplified = model.simplify()
        self.assertIsInstance(simplified, str)


class TestBaselineTrainer(unittest.TestCase):
    """Tests for baseline trainer implementation."""
    
    def test_trainer_creation(self):
        from pysr_baseline.model import SymbolicRegressor
        from pysr_baseline.trainer import SymbolicRegressionTrainer
        
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        trainer = SymbolicRegressionTrainer(model)
        
        self.assertIsNotNone(trainer.optimizer)
    
    def test_trainer_fit(self):
        from pysr_baseline.model import SymbolicRegressor
        from pysr_baseline.trainer import SymbolicRegressionTrainer
        
        model = SymbolicRegressor(n_features=2, max_depth=2, n_candidates=3)
        trainer = SymbolicRegressionTrainer(model)
        
        x = torch.randn(50, 2)
        y = x[:, 0] + x[:, 1]
        
        history = trainer.fit(x, y, n_epochs=10, verbose=False)
        
        self.assertGreater(len(history), 0)
        self.assertIn("mse", history[0])
    
    def test_trainer_evaluate(self):
        from pysr_baseline.model import SymbolicRegressor
        from pysr_baseline.trainer import SymbolicRegressionTrainer
        
        model = SymbolicRegressor(n_features=2, max_depth=2, n_candidates=3)
        trainer = SymbolicRegressionTrainer(model)
        
        x = torch.randn(50, 2)
        y = x[:, 0] + x[:, 1]
        
        metrics = trainer.evaluate(x, y)
        
        self.assertIn("mse", metrics)
        self.assertIn("r2", metrics)


class TestDPMemoization(unittest.TestCase):
    """Tests for DP memoization enhancement."""
    
    def test_expression_cache(self):
        from enhancements.dp_memoization import ExpressionCache
        cache = ExpressionCache(capacity=10)
        
        x = torch.randn(5, 3)
        result = torch.randn(5)
        
        cache.put(node_id=1, x=x, result=result)
        cached = cache.get(node_id=1, x=x)
        
        self.assertIsNotNone(cached)
        self.assertTrue(torch.allclose(cached, result))
    
    def test_cache_miss(self):
        from enhancements.dp_memoization import ExpressionCache
        cache = ExpressionCache(capacity=10)
        
        x = torch.randn(5, 3)
        cached = cache.get(node_id=999, x=x)
        
        self.assertIsNone(cached)
    
    def test_cache_eviction(self):
        from enhancements.dp_memoization import ExpressionCache
        cache = ExpressionCache(capacity=3)
        
        for i in range(5):
            x = torch.randn(5, 3)
            result = torch.randn(5)
            cache.put(node_id=i, x=x, result=result)
        
        stats = cache.get_stats()
        self.assertLessEqual(stats["size"], 3)
    
    def test_subproblem_table(self):
        from enhancements.dp_memoization import SubproblemTable
        table = SubproblemTable(max_complexity=10)
        
        table.put(complexity=3, structure_hash="abc", expression="test", fitness=0.5)
        result = table.get(complexity=3, structure_hash="abc")
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "test")
        self.assertEqual(result[1], 0.5)
    
    def test_structure_hasher(self):
        from enhancements.dp_memoization import StructureHasher
        from pysr_baseline.nodes import BinaryOpNode, WeightedInputNode
        
        left = WeightedInputNode(n_features=3)
        right = WeightedInputNode(n_features=3)
        node = BinaryOpNode(left, right)
        
        hash1 = StructureHasher.hash_structure(node)
        hash2 = StructureHasher.hash_structure(node)
        
        self.assertEqual(hash1, hash2)
    
    def test_memoized_evaluator(self):
        from enhancements.dp_memoization import MemoizedEvaluator
        from pysr_baseline.model import SymbolicRegressor
        
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=2)
        evaluator = MemoizedEvaluator(cache_capacity=100)
        
        x = torch.randn(10, 3)
        
        result1 = evaluator.evaluate(model, x)
        result2 = evaluator.evaluate(model, x)
        
        self.assertTrue(torch.allclose(result1, result2))
        
        stats = evaluator.get_stats()
        self.assertGreater(stats["hits"], 0)


class TestHybridOptimization(unittest.TestCase):
    """Tests for hybrid optimization enhancement."""
    
    def test_individual_creation(self):
        from enhancements.hybrid_optimization import Individual
        from pysr_baseline.model import SymbolicRegressor
        
        model = SymbolicRegressor(n_features=3, max_depth=2)
        ind = Individual(model=model, fitness=0.5)
        
        self.assertEqual(ind.fitness, 0.5)
    
    def test_individual_clone(self):
        from enhancements.hybrid_optimization import Individual
        from pysr_baseline.model import SymbolicRegressor
        
        model = SymbolicRegressor(n_features=3, max_depth=2)
        ind = Individual(model=model, fitness=0.5)
        clone = ind.clone()
        
        self.assertIsNot(clone.model, ind.model)
        self.assertEqual(clone.fitness, ind.fitness)
    
    def test_population(self):
        from enhancements.hybrid_optimization import Individual, Population
        from pysr_baseline.model import SymbolicRegressor
        
        pop = Population(max_size=10)
        
        for i in range(5):
            model = SymbolicRegressor(n_features=3, max_depth=2)
            ind = Individual(model=model, fitness=float(i))
            pop.add(ind)
        
        self.assertEqual(len(pop.individuals), 5)
        
        best = pop.get_best(1)
        self.assertEqual(best[0].fitness, 0.0)
    
    def test_mutation_operator(self):
        from enhancements.hybrid_optimization import Individual, MutationOperator
        from pysr_baseline.model import SymbolicRegressor
        
        model = SymbolicRegressor(n_features=3, max_depth=2)
        ind = Individual(model=model)
        
        mutated = MutationOperator.mutate_parameters(ind, strength=0.1)
        
        self.assertIsNot(mutated.model, ind.model)
    
    def test_crossover_operator(self):
        from enhancements.hybrid_optimization import Individual, CrossoverOperator
        from pysr_baseline.model import SymbolicRegressor
        
        model1 = SymbolicRegressor(n_features=3, max_depth=2)
        model2 = SymbolicRegressor(n_features=3, max_depth=2)
        
        parent1 = Individual(model=model1)
        parent2 = Individual(model=model2)
        
        child1, child2 = CrossoverOperator.parameter_crossover(parent1, parent2)
        
        self.assertIsNot(child1.model, parent1.model)
        self.assertIsNot(child2.model, parent2.model)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_baseline_end_to_end(self):
        from pysr_baseline.model import SymbolicRegressor
        from pysr_baseline.trainer import SymbolicRegressionTrainer
        
        torch.manual_seed(42)
        x = torch.linspace(-2, 2, 50).unsqueeze(1)
        y = x.squeeze() ** 2
        
        model = SymbolicRegressor(n_features=1, max_depth=2, n_candidates=3)
        trainer = SymbolicRegressionTrainer(model, complexity_weight=0.01)
        
        trainer.fit(x, y, n_epochs=50, verbose=False)
        metrics = trainer.evaluate(x, y)
        
        self.assertLess(metrics["mse"], 10.0)
    
    def test_dp_optimizer_integration(self):
        from pysr_baseline.model import SymbolicRegressor
        from enhancements.dp_memoization import OptimizedSymbolicTrainer
        
        torch.manual_seed(42)
        x = torch.linspace(-2, 2, 50).unsqueeze(1)
        y = x.squeeze() ** 2
        
        model = SymbolicRegressor(n_features=1, max_depth=2, n_candidates=3)
        trainer = OptimizedSymbolicTrainer(model, complexity_weight=0.01)
        
        history = trainer.fit(x, y, n_epochs=50, verbose=False)
        
        self.assertGreater(len(history), 0)


class TestRegression(unittest.TestCase):
    """Regression tests for consistency."""
    
    def test_determinism(self):
        from pysr_baseline.model import SymbolicRegressor
        
        torch.manual_seed(42)
        model1 = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        
        torch.manual_seed(42)
        model2 = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        
        x = torch.randn(10, 3)
        
        output1 = model1(x)
        output2 = model2(x)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_model_serialization(self):
        from pysr_baseline.model import SymbolicRegressor
        
        model = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
        x = torch.randn(10, 3)
        
        output_before = model(x)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            
            model_loaded = SymbolicRegressor(n_features=3, max_depth=2, n_candidates=3)
            model_loaded.load_state_dict(torch.load(f.name, weights_only=True))
        
        output_after = model_loaded(x)
        
        self.assertTrue(torch.allclose(output_before, output_after))


class TestPerformance(unittest.TestCase):
    """Performance tests for speed and memory."""
    
    def test_forward_pass_speed(self):
        from pysr_baseline.model import SymbolicRegressor
        
        model = SymbolicRegressor(n_features=10, max_depth=3, n_candidates=5)
        x = torch.randn(1000, 10)
        
        for _ in range(5):
            _ = model(x)
        
        start = time.time()
        for _ in range(50):
            _ = model(x)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 10.0)
    
    def test_memoization_speedup(self):
        from pysr_baseline.model import SymbolicRegressor
        from enhancements.dp_memoization import benchmark_optimization
        
        model = SymbolicRegressor(n_features=5, max_depth=2, n_candidates=3)
        x = torch.randn(100, 5)
        y = torch.randn(100)
        
        results = benchmark_optimization(model, x, y, n_iterations=20)
        
        self.assertIn("speedup", results)


def run_all_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestBaselineOperators,
        TestBaselineNodes,
        TestBaselineModel,
        TestBaselineTrainer,
        TestDPMemoization,
        TestHybridOptimization,
        TestIntegration,
        TestRegression,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_all_tests()

"""
Hybrid Trainer Module
---------------------
Combines evolutionary search with gradient-based optimization.

The hybrid approach alternates between:
    1. Evolutionary phase: Explore structure space
    2. Gradient phase: Refine parameters of promising candidates
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import time

from .evolutionary import (
    Individual,
    Population,
    EvolutionaryOptimizer,
    MutationOperator
)


@dataclass
class HybridConfig:
    """Configuration for hybrid training."""
    population_size: int = 30
    n_generations: int = 20
    gradient_epochs: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    learning_rate: float = 0.01
    complexity_weight: float = 0.01
    n_cycles: int = 5
    elite_gradient_epochs: int = 100


class AdaptiveScheduler:
    """
    Adaptively schedules evolutionary vs gradient phases.
    
    Strategy:
        - Increase evolutionary exploration when stuck in local optima
        - Increase gradient refinement when good structures are found
    """
    
    def __init__(
        self,
        initial_evo_ratio: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        self.evo_ratio = initial_evo_ratio
        self.adaptation_rate = adaptation_rate
        self.improvement_history: List[float] = []
    
    def update(self, improvement: float) -> None:
        self.improvement_history.append(improvement)
        
        if len(self.improvement_history) >= 3:
            recent = self.improvement_history[-3:]
            avg_improvement = sum(recent) / len(recent)
            
            if avg_improvement < 0.001:
                self.evo_ratio = min(0.9, self.evo_ratio + self.adaptation_rate)
            else:
                self.evo_ratio = max(0.1, self.evo_ratio - self.adaptation_rate)
    
    def get_allocation(self, total_budget: int) -> Tuple[int, int]:
        evo_budget = int(total_budget * self.evo_ratio)
        grad_budget = total_budget - evo_budget
        return evo_budget, grad_budget


class HybridTrainer:
    """
    Hybrid trainer combining evolutionary and gradient optimization.
    
    Training Loop:
        For each cycle:
            1. Run evolutionary optimization for structure search
            2. Take top-k individuals
            3. Apply gradient descent to refine parameters
            4. Update population with improved individuals
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: Optional[HybridConfig] = None
    ):
        self.model_factory = model_factory
        self.config = config or HybridConfig()
        
        self.evo_optimizer = EvolutionaryOptimizer(
            population_size=self.config.population_size,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate
        )
        
        self.scheduler = AdaptiveScheduler()
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
    
    def _gradient_refine(
        self,
        individual: Individual,
        x: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int
    ) -> Individual:
        model = individual.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        best_loss = float('inf')
        best_state = None
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            y_pred = model(x)
            mse = torch.mean((y_pred - y) ** 2)
            complexity = model.get_complexity()
            loss = mse + self.config.complexity_weight * complexity
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        with torch.no_grad():
            y_pred = model(x)
            individual.fitness = torch.mean((y_pred - y) ** 2).item()
            individual.complexity = model.get_complexity().item()
        
        return individual
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        verbose: bool = True
    ) -> nn.Module:
        if y.dim() == 2:
            y = y.squeeze(-1)
        
        self.evo_optimizer.initialize(self.model_factory, x, y)
        
        start_time = time.time()
        
        for cycle in range(self.config.n_cycles):
            cycle_start = time.time()
            
            for gen in range(self.config.n_generations):
                self.evo_optimizer.evolve_generation(x, y)
            
            top_individuals = self.evo_optimizer.population.get_best(5)
            
            for ind in top_individuals:
                self._gradient_refine(ind, x, y, self.config.gradient_epochs)
            
            for ind in top_individuals:
                self.evo_optimizer.population.add(ind)
            
            best = self.evo_optimizer.get_best()
            if self.best_individual is None or best.fitness < self.best_individual.fitness:
                self.best_individual = best.clone()
            
            cycle_time = time.time() - cycle_start
            stats = self.evo_optimizer.population.get_statistics()
            stats["cycle"] = cycle
            stats["cycle_time"] = cycle_time
            self.history.append(stats)
            
            if verbose:
                print(f"Cycle {cycle+1}/{self.config.n_cycles} | "
                      f"Best: {stats['min_fitness']:.6f} | "
                      f"Avg: {stats['avg_fitness']:.6f} | "
                      f"Time: {cycle_time:.1f}s")
        
        self._gradient_refine(
            self.best_individual, x, y, 
            self.config.elite_gradient_epochs
        )
        
        total_time = time.time() - start_time
        if verbose:
            print(f"Training complete | Total time: {total_time:.1f}s | "
                  f"Final MSE: {self.best_individual.fitness:.6f}")
        
        return self.best_individual.model
    
    def get_best_model(self) -> nn.Module:
        if self.best_individual is None:
            raise RuntimeError("No training performed yet")
        return self.best_individual.model
    
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

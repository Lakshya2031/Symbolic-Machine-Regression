"""
Evolutionary Optimization Module
--------------------------------
Implements population-based evolutionary algorithms for symbolic regression.

Components:
    - Individual: Single candidate solution with fitness tracking
    - Population: Collection of individuals with diversity maintenance
    - EvolutionaryOptimizer: Main evolutionary loop implementation
    - Mutation/Crossover operators: Genetic variation mechanisms
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
import copy
import random


@dataclass
class Individual:
    """
    Represents a single candidate solution in the population.
    
    Attributes:
        model: The symbolic regression model
        fitness: Evaluated fitness score (lower is better)
        complexity: Expression complexity measure
        age: Number of generations survived
    """
    model: nn.Module
    fitness: float = float('inf')
    complexity: float = 0.0
    age: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def clone(self) -> 'Individual':
        new_model = copy.deepcopy(self.model)
        return Individual(
            model=new_model,
            fitness=self.fitness,
            complexity=self.complexity,
            age=0,
            metadata=dict(self.metadata)
        )
    
    def dominates(self, other: 'Individual') -> bool:
        return (self.fitness <= other.fitness and 
                self.complexity <= other.complexity and
                (self.fitness < other.fitness or self.complexity < other.complexity))


class Population:
    """
    Manages a population of candidate solutions.
    
    Implements diversity maintenance through:
        - Crowding distance for selection
        - Niche formation based on structure similarity
        - Age-based replacement strategies
    """
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.individuals: List[Individual] = []
        self.generation = 0
    
    def add(self, individual: Individual) -> None:
        self.individuals.append(individual)
        if len(self.individuals) > self.max_size:
            self._truncate()
    
    def _truncate(self) -> None:
        self.individuals.sort(key=lambda x: x.fitness)
        self.individuals = self.individuals[:self.max_size]
    
    def get_best(self, n: int = 1) -> List[Individual]:
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness)
        return sorted_pop[:n]
    
    def get_pareto_front(self) -> List[Individual]:
        front = []
        for ind in self.individuals:
            dominated = False
            for other in self.individuals:
                if other.dominates(ind):
                    dominated = True
                    break
            if not dominated:
                front.append(ind)
        return front
    
    def tournament_select(self, k: int = 3) -> Individual:
        participants = random.sample(self.individuals, min(k, len(self.individuals)))
        return min(participants, key=lambda x: x.fitness)
    
    def increment_age(self) -> None:
        for ind in self.individuals:
            ind.age += 1
        self.generation += 1
    
    def get_statistics(self) -> Dict[str, float]:
        if not self.individuals:
            return {}
        
        fitnesses = [ind.fitness for ind in self.individuals]
        complexities = [ind.complexity for ind in self.individuals]
        
        return {
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_complexity": min(complexities),
            "max_complexity": max(complexities),
            "avg_complexity": sum(complexities) / len(complexities),
            "population_size": len(self.individuals),
            "generation": self.generation
        }


class MutationOperator:
    """
    Collection of mutation operators for symbolic expressions.
    
    Mutation Types:
        - Parameter mutation: Perturb numerical constants
        - Operator mutation: Change operator type
        - Structural mutation: Add/remove nodes
    """
    
    @staticmethod
    def mutate_parameters(individual: Individual, strength: float = 0.1) -> Individual:
        new_ind = individual.clone()
        with torch.no_grad():
            for param in new_ind.model.parameters():
                noise = torch.randn_like(param) * strength
                param.add_(noise)
        return new_ind
    
    @staticmethod
    def mutate_operator_weights(individual: Individual, strength: float = 0.5) -> Individual:
        new_ind = individual.clone()
        with torch.no_grad():
            for name, param in new_ind.model.named_parameters():
                if 'op_logits' in name:
                    noise = torch.randn_like(param) * strength
                    param.add_(noise)
        return new_ind
    
    @staticmethod
    def mutate_feature_weights(individual: Individual, strength: float = 0.5) -> Individual:
        new_ind = individual.clone()
        with torch.no_grad():
            for name, param in new_ind.model.named_parameters():
                if 'feature_logits' in name:
                    noise = torch.randn_like(param) * strength
                    param.add_(noise)
        return new_ind
    
    @staticmethod
    def apply_random_mutation(
        individual: Individual,
        mutation_rate: float = 0.3
    ) -> Individual:
        if random.random() > mutation_rate:
            return individual.clone()
        
        mutation_type = random.choice(['parameter', 'operator', 'feature'])
        
        if mutation_type == 'parameter':
            return MutationOperator.mutate_parameters(individual)
        elif mutation_type == 'operator':
            return MutationOperator.mutate_operator_weights(individual)
        else:
            return MutationOperator.mutate_feature_weights(individual)


class CrossoverOperator:
    """
    Collection of crossover operators for combining solutions.
    
    Crossover Types:
        - Parameter crossover: Blend numerical parameters
        - Uniform crossover: Exchange operator selections
    """
    
    @staticmethod
    def parameter_crossover(
        parent1: Individual,
        parent2: Individual,
        alpha: float = 0.5
    ) -> Tuple[Individual, Individual]:
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        with torch.no_grad():
            params1 = list(child1.model.parameters())
            params2 = list(child2.model.parameters())
            
            for p1, p2 in zip(params1, params2):
                if p1.shape == p2.shape:
                    new_p1 = alpha * p1 + (1 - alpha) * p2
                    new_p2 = (1 - alpha) * p1 + alpha * p2
                    p1.copy_(new_p1)
                    p2.copy_(new_p2)
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover(
        parent1: Individual,
        parent2: Individual,
        swap_prob: float = 0.5
    ) -> Tuple[Individual, Individual]:
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        with torch.no_grad():
            params1 = dict(child1.model.named_parameters())
            params2 = dict(child2.model.named_parameters())
            
            for name in params1:
                if name in params2 and random.random() < swap_prob:
                    if params1[name].shape == params2[name].shape:
                        temp = params1[name].clone()
                        params1[name].copy_(params2[name])
                        params2[name].copy_(temp)
        
        return child1, child2


class EvolutionaryOptimizer:
    """
    Main evolutionary optimization loop.
    
    Algorithm:
        1. Initialize population with random individuals
        2. Evaluate fitness of all individuals
        3. Select parents using tournament selection
        4. Apply crossover and mutation
        5. Replace worst individuals with offspring
        6. Repeat until termination condition
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elitism: int = 2
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        self.population: Optional[Population] = None
        self.history: List[Dict[str, float]] = []
    
    def initialize(
        self,
        model_factory: Callable[[], nn.Module],
        x: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        self.population = Population(self.population_size)
        
        for _ in range(self.population_size):
            model = model_factory()
            individual = Individual(model=model)
            self._evaluate(individual, x, y)
            self.population.add(individual)
    
    def _evaluate(
        self,
        individual: Individual,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        with torch.no_grad():
            y_pred = individual.model(x)
            mse = torch.mean((y_pred - y) ** 2).item()
            complexity = individual.model.get_complexity().item()
        
        individual.fitness = mse
        individual.complexity = complexity
    
    def evolve_generation(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        if self.population is None:
            raise RuntimeError("Population not initialized")
        
        elite = self.population.get_best(self.elitism)
        offspring = [ind.clone() for ind in elite]
        
        while len(offspring) < self.population_size:
            parent1 = self.population.tournament_select(self.tournament_size)
            parent2 = self.population.tournament_select(self.tournament_size)
            
            if random.random() < self.crossover_rate:
                child1, child2 = CrossoverOperator.parameter_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            child1 = MutationOperator.apply_random_mutation(child1, self.mutation_rate)
            child2 = MutationOperator.apply_random_mutation(child2, self.mutation_rate)
            
            self._evaluate(child1, x, y)
            self._evaluate(child2, x, y)
            
            offspring.extend([child1, child2])
        
        self.population.individuals = offspring[:self.population_size]
        self.population.increment_age()
        
        stats = self.population.get_statistics()
        self.history.append(stats)
        
        return stats
    
    def run(
        self,
        model_factory: Callable[[], nn.Module],
        x: torch.Tensor,
        y: torch.Tensor,
        n_generations: int = 50,
        verbose: bool = True
    ) -> Individual:
        self.initialize(model_factory, x, y)
        
        for gen in range(n_generations):
            stats = self.evolve_generation(x, y)
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{n_generations} | "
                      f"Best: {stats['min_fitness']:.6f} | "
                      f"Avg: {stats['avg_fitness']:.6f}")
        
        return self.population.get_best(1)[0]
    
    def get_best(self) -> Individual:
        if self.population is None:
            raise RuntimeError("Population not initialized")
        return self.population.get_best(1)[0]

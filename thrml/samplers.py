"""
Conditional samplers for different node types
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any
from .nodes import AbstractNode


class AbstractSampler(ABC):
    """Base class for conditional samplers"""
    
    @abstractmethod
    def sample(self, 
               rng: np.random.Generator,
               interactions: List[Any],
               neighbor_states: List[np.ndarray],
               n_nodes: int) -> np.ndarray:
        """
        Sample new states for a block of nodes.
        
        Args:
            rng: NumPy random generator
            interactions: List of interaction parameters
            neighbor_states: States of neighboring nodes
            n_nodes: Number of nodes to sample
        
        Returns:
            Array of new states
        """
        pass


class BernoulliSampler(AbstractSampler):
    """
    Sampler for spin nodes: P(s) ∝ exp(γ·s), s ∈ {-1, 1}
    Stores as bool: True=+1, False=-1
    """
    
    def compute_field(self, 
                      interactions: List[Any],
                      neighbor_states: List[np.ndarray],
                      n_nodes: int) -> np.ndarray:
        """
        Compute local field γ for each node.
        To be overridden by specific models.
        """
        return np.zeros(n_nodes)
    
    def sample(self, 
               rng: np.random.Generator,
               interactions: List[Any],
               neighbor_states: List[np.ndarray],
               n_nodes: int) -> np.ndarray:
        """Sample spins given local field"""
        gamma = self.compute_field(interactions, neighbor_states, n_nodes)
        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * gamma))
        return rng.random(n_nodes) < prob_plus  # True = +1, False = -1


class CategoricalSampler(AbstractSampler):
    """
    Sampler for categorical nodes: P(k) ∝ exp(θ_k)
    """
    
    def compute_logits(self,
                       interactions: List[Any],
                       neighbor_states: List[np.ndarray],
                       n_nodes: int,
                       n_categories: int) -> np.ndarray:
        """
        Compute logits θ for each node and category.
        Shape: (n_nodes, n_categories)
        """
        return np.zeros((n_nodes, n_categories))
    
    def sample(self,
               rng: np.random.Generator,
               interactions: List[Any],
               neighbor_states: List[np.ndarray],
               n_nodes: int,
               n_categories: int = 2) -> np.ndarray:
        """Sample categorical states given logits"""
        logits = self.compute_logits(interactions, neighbor_states, n_nodes, n_categories)
        
        # Softmax + categorical sampling
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Sample
        cumprobs = np.cumsum(probs, axis=1)
        u = rng.random((n_nodes, 1))
        samples = np.sum(u > cumprobs, axis=1)
        
        return samples.astype(np.uint8)


class IsingSampler(BernoulliSampler):
    """
    Ising model sampler: γ_i = Σ_j J_ij s_j + h_i
    """
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def compute_field(self,
                      interactions: List[Dict],
                      neighbor_states: List[np.ndarray],
                      n_nodes: int) -> np.ndarray:
        """
        Compute field: γ = β(J·s + h)
        
        interactions[0] should contain:
            - 'weights': coupling matrix J (n_nodes, n_neighbors)
            - 'biases': external field h (n_nodes,)
        neighbor_states[0]: neighbor spins (n_neighbors,)
        """
        if not interactions:
            return np.zeros(n_nodes)
        
        params = interactions[0]
        weights = params.get('weights', np.zeros((n_nodes, 0)))
        biases = params.get('biases', np.zeros(n_nodes))
        
        # Convert bool to {-1, +1}
        if neighbor_states and len(neighbor_states[0]) > 0:
            spins = neighbor_states[0].astype(np.float32) * 2 - 1
            field = np.sum(weights * spins, axis=1) + biases
        else:
            field = biases
        
        return self.beta * field

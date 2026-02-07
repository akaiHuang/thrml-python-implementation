"""
Pre-built models
"""

import numpy as np
from typing import List, Optional, Tuple
from .nodes import SpinNode
from .blocks import Block
from .block_sampling import BlockGibbsSpec, BlockSamplingProgram
from .samplers import IsingSampler


class IsingModel:
    """
    Ising model on arbitrary graph.
    
    Energy: E = -Σ_<i,j> J_ij s_i s_j - Σ_i h_i s_i
    """
    
    def __init__(self,
                 nodes: List[SpinNode],
                 edges: List[Tuple[int, int]],
                 biases: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 beta: float = 1.0):
        """
        Args:
            nodes: List of spin nodes
            edges: List of (i, j) edge indices
            biases: External fields h (default: zeros)
            weights: Coupling strengths J (default: ones)
            beta: Inverse temperature
        """
        self.nodes = nodes
        self.edges = edges
        self.beta = beta
        
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        if biases is None:
            self.biases = np.zeros(n_nodes)
        else:
            self.biases = np.asarray(biases)
        
        if weights is None:
            self.weights = np.ones(n_edges)
        else:
            self.weights = np.asarray(weights)
        
        # Build adjacency structure
        self.neighbors = [[] for _ in range(n_nodes)]
        self.edge_weights = [[] for _ in range(n_nodes)]
        
        for (i, j), w in zip(edges, self.weights):
            self.neighbors[i].append(j)
            self.edge_weights[i].append(w)
            self.neighbors[j].append(i)
            self.edge_weights[j].append(w)
    
    def create_sampling_program(self, coloring: Optional[List[int]] = None) -> BlockSamplingProgram:
        """
        Create block sampling program.
        
        Args:
            coloring: Graph coloring (default: sequential)
        
        Returns:
            BlockSamplingProgram ready for sampling
        """
        n_nodes = len(self.nodes)
        
        if coloring is None:
            # Sequential: each node is its own block
            coloring = list(range(n_nodes))
        
        # Group nodes by color
        color_groups = {}
        for node_idx, color in enumerate(coloring):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(node_idx)
        
        # Create blocks
        free_blocks = []
        for color in sorted(color_groups.keys()):
            node_indices = color_groups[color]
            block_nodes = [self.nodes[i] for i in node_indices]
            free_blocks.append(Block(block_nodes))
        
        # Create samplers and interactions
        samplers = []
        interactions = []
        
        for block in free_blocks:
            # Build interaction matrix for this block
            block_size = len(block)
            
            # Collect neighbor information
            max_neighbors = max(len(self.neighbors[i]) for i in range(n_nodes))
            weight_matrix = np.zeros((block_size, max_neighbors))
            
            for local_idx, node in enumerate(block.nodes):
                node_idx = self.nodes.index(node)
                for k, (neighbor_idx, w) in enumerate(zip(self.neighbors[node_idx], self.edge_weights[node_idx])):
                    if k < max_neighbors:
                        weight_matrix[local_idx, k] = w
            
            # Block biases
            block_biases = np.array([self.biases[self.nodes.index(node)] for node in block.nodes])
            
            interactions.append({
                'weights': weight_matrix,
                'biases': block_biases
            })
            
            samplers.append(IsingSampler(beta=self.beta))
        
        gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks=[])
        
        return BlockSamplingProgram(gibbs_spec, samplers, interactions)
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute energy of a configuration.
        
        Args:
            state: Spin configuration (bool array: True=+1, False=-1)
        
        Returns:
            Energy value
        """
        spins = state.astype(np.float32) * 2 - 1  # bool -> {-1, +1}
        
        E = 0.0
        # Edge terms
        for (i, j), w in zip(self.edges, self.weights):
            E -= w * spins[i] * spins[j]
        
        # Field terms
        E -= np.sum(self.biases * spins)
        
        return E
    
    def magnetization(self, state: np.ndarray) -> float:
        """Average magnetization"""
        spins = state.astype(np.float32) * 2 - 1
        return np.mean(spins)

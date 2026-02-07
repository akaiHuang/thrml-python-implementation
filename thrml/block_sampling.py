"""
Block Gibbs sampling engine - Pure Python/NumPy
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from .blocks import Block, BlockSpec
from .nodes import AbstractNode
from .samplers import AbstractSampler


class BlockGibbsSpec(BlockSpec):
    """
    Extended BlockSpec with free/clamped blocks and sampling order.
    """
    
    def __init__(self,
                 free_blocks: List[Block],
                 clamped_blocks: List[Block],
                 sampling_order: Optional[List[List[int]]] = None):
        """
        Args:
            free_blocks: Blocks to be sampled
            clamped_blocks: Blocks held fixed
            sampling_order: Groups of block indices to sample together
                          (default: sample each block sequentially)
        """
        super().__init__(free_blocks + clamped_blocks)
        
        self.free_blocks = free_blocks
        self.clamped_blocks = clamped_blocks
        
        if sampling_order is None:
            # Sequential sampling: [[0], [1], [2], ...]
            self.sampling_order = [[i] for i in range(len(free_blocks))]
        else:
            self.sampling_order = sampling_order
    
    def __repr__(self):
        return f"BlockGibbsSpec(free={len(self.free_blocks)}, clamped={len(self.clamped_blocks)})"


class BlockSamplingProgram:
    """
    Complete sampling program for a PGM.
    """
    
    def __init__(self,
                 gibbs_spec: BlockGibbsSpec,
                 samplers: List[AbstractSampler],
                 interactions: List[Dict[str, Any]]):
        """
        Args:
            gibbs_spec: Block structure
            samplers: One sampler per free block
            interactions: Interaction parameters for each block
        """
        if len(samplers) != len(gibbs_spec.free_blocks):
            raise ValueError(f"Need {len(gibbs_spec.free_blocks)} samplers, got {len(samplers)}")
        
        self.gibbs_spec = gibbs_spec
        self.samplers = samplers
        self.interactions = interactions
    
    def __repr__(self):
        return f"BlockSamplingProgram({self.gibbs_spec})"


def sample_single_block(rng: np.random.Generator,
                        block_idx: int,
                        free_states: List[np.ndarray],
                        clamped_states: List[np.ndarray],
                        program: BlockSamplingProgram) -> np.ndarray:
    """
    Sample one block given current states.
    
    Args:
        rng: Random number generator
        block_idx: Index of block to sample
        free_states: Current states of free blocks
        clamped_states: Fixed states of clamped blocks
        program: Sampling program
    
    Returns:
        New state for the block
    """
    block = program.gibbs_spec.free_blocks[block_idx]
    sampler = program.samplers[block_idx]
    
    # Get global state
    all_states = free_states + clamped_states
    global_state = program.gibbs_spec.block_to_global(all_states)
    
    # Extract neighbor states (simplified: just pass global state)
    # In real implementation, would slice based on graph structure
    neighbor_states = [global_state[block.node_type]]
    
    # Get interaction parameters for this block
    block_interactions = [program.interactions[block_idx]] if block_idx < len(program.interactions) else []
    
    # Sample
    new_state = sampler.sample(rng, block_interactions, neighbor_states, len(block))
    
    return new_state


def sample_blocks(rng: np.random.Generator,
                  free_states: List[np.ndarray],
                  clamped_states: List[np.ndarray],
                  program: BlockSamplingProgram) -> List[np.ndarray]:
    """
    Perform one full iteration of block Gibbs sampling.
    
    Args:
        rng: Random number generator
        free_states: Current states of free blocks
        clamped_states: Fixed states of clamped blocks
        program: Sampling program
    
    Returns:
        Updated states of free blocks
    """
    new_free_states = list(free_states)  # Copy
    
    # Sample according to sampling order
    for group in program.gibbs_spec.sampling_order:
        # In true parallel implementation, these would be sampled simultaneously
        # For now, sample sequentially within group
        for block_idx in group:
            new_free_states[block_idx] = sample_single_block(
                rng, block_idx, new_free_states, clamped_states, program
            )
    
    return new_free_states


def sample_states(rng: np.random.Generator,
                  program: BlockSamplingProgram,
                  init_free_states: List[np.ndarray],
                  clamped_states: List[np.ndarray],
                  n_warmup: int = 100,
                  n_samples: int = 1000,
                  thin: int = 1) -> List[np.ndarray]:
    """
    Run full sampling chain with warmup and thinning.
    
    Args:
        rng: Random number generator
        program: Sampling program
        init_free_states: Initial states
        clamped_states: Fixed states
        n_warmup: Warmup steps
        n_samples: Number of samples to collect
        thin: Steps between samples
    
    Returns:
        List of sample arrays, one per free block
    """
    current_states = list(init_free_states)
    
    # Warmup
    for _ in range(n_warmup):
        current_states = sample_blocks(rng, current_states, clamped_states, program)
    
    # Collect samples
    samples = [[] for _ in range(len(program.gibbs_spec.free_blocks))]
    
    for _ in range(n_samples):
        # Thin
        for _ in range(thin):
            current_states = sample_blocks(rng, current_states, clamped_states, program)
        
        # Record
        for i, state in enumerate(current_states):
            samples[i].append(state.copy())
    
    # Stack samples
    return [np.stack(s, axis=0) for s in samples]

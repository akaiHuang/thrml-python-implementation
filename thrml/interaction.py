"""
Interaction definitions for PGMs
"""

from typing import List, Dict, Any
from .blocks import Block
from .nodes import AbstractNode


class Interaction:
    """
    Defines how a set of nodes interact.
    """
    
    def __init__(self, 
                 head_nodes: List[AbstractNode],
                 tail_nodes: List[AbstractNode],
                 parameters: Dict[str, Any]):
        """
        Args:
            head_nodes: Nodes being updated
            tail_nodes: Nodes providing context
            parameters: Interaction parameters (weights, biases, etc.)
        """
        self.head_nodes = head_nodes
        self.tail_nodes = tail_nodes
        self.parameters = parameters
    
    def __repr__(self):
        return f"Interaction({len(self.head_nodes)} head, {len(self.tail_nodes)} tail)"


class InteractionGroup:
    """
    Groups multiple interactions with same structure.
    """
    
    def __init__(self,
                 head_block: Block,
                 tail_blocks: List[Block],
                 interaction_fn: Any):
        """
        Args:
            head_block: Block of nodes being sampled
            tail_blocks: Blocks providing neighbor information
            interaction_fn: Function defining the interaction
        """
        self.head_block = head_block
        self.tail_blocks = tail_blocks
        self.interaction_fn = interaction_fn
    
    def __repr__(self):
        return f"InteractionGroup(head={len(self.head_block)}, tails={[len(b) for b in self.tail_blocks]})"

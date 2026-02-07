"""
Block management for efficient sampling
"""

import numpy as np
from typing import List, Dict, Tuple, Type, Sequence
from .nodes import AbstractNode, SpinNode, CategoricalNode


class Block:
    """
    Collection of nodes that can be sampled simultaneously.
    All nodes in a block must be of the same type.
    """
    
    def __init__(self, nodes: Sequence[AbstractNode]):
        self.nodes = tuple(nodes)
        if self.nodes:
            first_type = type(self.nodes[0])
            if any(type(node) != first_type for node in self.nodes):
                raise ValueError("All nodes in a block must be of the same type")
    
    @property
    def node_type(self) -> Type[AbstractNode]:
        if not self.nodes:
            raise ValueError("Block is empty")
        return type(self.nodes[0])
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __repr__(self):
        return f"Block({len(self)} nodes, type={self.node_type.__name__})"


class BlockSpec:
    """
    Mapping between block states and global states.
    
    Block state: list of arrays, one per block
    Global state: dict mapping node types to concatenated arrays
    """
    
    def __init__(self, blocks: List[Block]):
        self.blocks = blocks
        
        # Build node -> (block_idx, position_in_block) mapping
        self.node_to_location: Dict[AbstractNode, Tuple[int, int]] = {}
        for block_idx, block in enumerate(blocks):
            for pos, node in enumerate(block.nodes):
                if node in self.node_to_location:
                    raise RuntimeError("Node appears twice in blocks")
                self.node_to_location[node] = (block_idx, pos)
        
        # Group blocks by node type
        self.blocks_by_type: Dict[Type[AbstractNode], List[int]] = {}
        for block_idx, block in enumerate(blocks):
            if len(block) == 0:
                continue
            node_type = block.node_type
            if node_type not in self.blocks_by_type:
                self.blocks_by_type[node_type] = []
            self.blocks_by_type[node_type].append(block_idx)
    
    def block_to_global(self, block_states: List[np.ndarray]) -> Dict[Type[AbstractNode], np.ndarray]:
        """Convert block states to global state"""
        global_state = {}
        
        for node_type, block_indices in self.blocks_by_type.items():
            arrays = [block_states[i] for i in block_indices]
            global_state[node_type] = np.concatenate(arrays, axis=0)
        
        return global_state
    
    def global_to_block(self, global_state: Dict[Type[AbstractNode], np.ndarray]) -> List[np.ndarray]:
        """Convert global state to block states"""
        block_states = []
        
        for block in self.blocks:
            if len(block) == 0:
                block_states.append(np.array([]))
                continue
            
            node_type = block.node_type
            
            # Find positions of this block's nodes in global array
            positions = []
            for node in block.nodes:
                # Count how many nodes of same type come before this node
                pos = 0
                for other_block_idx, other_block in enumerate(self.blocks):
                    if other_block_idx == self.blocks.index(block):
                        pos += list(block.nodes).index(node)
                        break
                    if len(other_block) > 0 and other_block.node_type == node_type:
                        pos += len(other_block)
                positions.append(pos)
            
            block_states.append(global_state[node_type][positions])
        
        return block_states
    
    def get_node_positions(self, nodes: Block) -> Tuple[Type[AbstractNode], np.ndarray]:
        """Get global positions of nodes in a block"""
        if len(nodes) == 0:
            raise ValueError("Empty block")
        
        node_type = nodes.node_type
        positions = []
        
        for node in nodes:
            pos = 0
            found = False
            for block in self.blocks:
                if len(block) == 0 or block.node_type != node_type:
                    continue
                if node in block.nodes:
                    pos += list(block.nodes).index(node)
                    found = True
                    break
                pos += len(block)
            
            if not found:
                raise ValueError(f"Node {node} not found in blocks")
            positions.append(pos)
        
        return node_type, np.array(positions, dtype=np.int32)

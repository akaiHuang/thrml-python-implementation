"""
THRML-PY: Pure Python + NumPy implementation of THRML
No JAX, No GPU dependencies - CPU optimized
"""

from .nodes import SpinNode, CategoricalNode, AbstractNode
from .blocks import Block, BlockSpec
from .samplers import BernoulliSampler, CategoricalSampler
from .block_sampling import BlockGibbsSpec, BlockSamplingProgram, sample_blocks, sample_states
from .interaction import Interaction, InteractionGroup
from .models import IsingModel

__version__ = "0.1.0"

__all__ = [
    "SpinNode",
    "CategoricalNode",
    "AbstractNode",
    "Block",
    "BlockSpec",
    "BernoulliSampler",
    "CategoricalSampler",
    "BlockGibbsSpec",
    "BlockSamplingProgram",
    "sample_blocks",
    "sample_states",
    "Interaction",
    "InteractionGroup",
    "IsingModel",
]

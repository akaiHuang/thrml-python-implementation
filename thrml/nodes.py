"""
Node types for probabilistic graphical models
"""

from typing import ClassVar
from dataclasses import dataclass


class _UniqueID:
    """Ensures unique identifier for each node instance"""
    
    _counter: ClassVar[int] = 0
    
    def __init__(self):
        self._hash = _UniqueID._counter
        _UniqueID._counter += 1
    
    def __eq__(self, other):
        if not isinstance(other, _UniqueID):
            return False
        return self._hash == other._hash
    
    def __hash__(self):
        return self._hash
    
    def __lt__(self, other):
        if isinstance(other, _UniqueID):
            return self._hash < other._hash
        raise RuntimeError("less than only defined between _UniqueIDs")


@dataclass(eq=False)
class AbstractNode(_UniqueID):
    """Base class for all PGM nodes"""
    
    def __post_init__(self):
        _UniqueID.__init__(self)
    
    def __new__(cls, *args, **kwargs):
        if cls is AbstractNode:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)


class SpinNode(AbstractNode):
    """Spin variable: state in {-1, 1} (stored as bool: True=1, False=-1)"""
    pass


class CategoricalNode(AbstractNode):
    """Categorical variable: state in {0, 1, ..., K-1}"""
    pass

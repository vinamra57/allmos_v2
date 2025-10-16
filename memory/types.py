"""
Abstract base classes for memory management.

Design Philosophy:
- Clean interface for block allocation/deallocation
- Support for prefix caching through hash-based lookups
"""
from abc import ABC, abstractmethod


class BlockManager(ABC):
    """
    Abstract block manager for KV cache memory.

    Responsibilities:
    - Allocate and deallocate memory blocks
    - Track block usage and reference counts
    - Support prefix caching for shared prompt prefixes
    """

    @abstractmethod
    def can_allocate(self, seq) -> bool:
        """Check if enough blocks are available for sequence."""
        pass

    @abstractmethod
    def allocate(self, seq) -> None:
        """Allocate blocks for sequence, reusing cached blocks if possible."""
        pass

    @abstractmethod
    def deallocate(self, seq) -> None:
        """Deallocate blocks used by sequence."""
        pass

    @abstractmethod
    def can_append(self, seq) -> bool:
        """Check if we can append a token to sequence (need new block if full)."""
        pass

    @abstractmethod
    def may_append(self, seq) -> None:
        """
        Prepare for appending a token to sequence.
        May allocate a new block if current block is full.
        """
        pass

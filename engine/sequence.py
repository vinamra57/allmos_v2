"""
Sequence management for tracking generation state.

Design Philosophy:
- Immutable prompt tokens, mutable completion tokens
- Block-based organization for efficient memory management
- Support prefix caching through hash-based block identification

Optimization Notes:
- Block size set to 256 (matching nano-vLLM) for Flash Attention compatibility
- Separate tracking of cached vs uncached tokens for prefix caching
"""
from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List

from sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Execution status of a sequence."""
    WAITING = auto()    # In queue, not yet scheduled
    RUNNING = auto()    # Currently being processed
    FINISHED = auto()   # Generation complete


class Sequence:
    """
    Represents a single generation request with its state.

    Attributes:
        seq_id: Unique sequence identifier
        status: Current execution status
        token_ids: All tokens (prompt + completion)
        block_table: List of KV cache block IDs
        num_cached_tokens: Number of tokens with cached KV states
        sampling_params: Generation parameters
    """

    # Class-level configuration and counter
    block_size: int = 256  # Must match config.kvcache_block_size
    _counter = count()

    def __init__(self, token_ids: List[int], sampling_params: SamplingParams = None):
        """
        Initialize a new sequence.

        Args:
            token_ids: Initial prompt token IDs
            sampling_params: Generation parameters (default if None)
        """
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING

        # Token management
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)

        # KV cache management
        self.num_cached_tokens = 0
        self.block_table: List[int] = []

        # Sampling configuration
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        """Total number of tokens in sequence."""
        return self.num_tokens

    def __getitem__(self, key):
        """Access token IDs by index or slice."""
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        """Check if sequence generation is complete."""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """Number of generated tokens (excluding prompt)."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> List[int]:
        """Get prompt token IDs."""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> List[int]:
        """Get completion token IDs."""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """Number of blocks with cached KV states."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """Total number of blocks needed for all tokens."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """Number of tokens in the last (potentially partial) block."""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> List[int]:
        """
        Get tokens in block i.

        Args:
            i: Block index

        Returns:
            List of token IDs in the block (up to block_size tokens)
        """
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        start = i * self.block_size
        end = min(start + self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        """
        Append a generated token to the sequence.

        Args:
            token_id: New token ID to append
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        Serialize sequence for inter-process communication.

        Optimization: Only send last_token for decode phase (not full token list).
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        """Deserialize sequence from inter-process communication."""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

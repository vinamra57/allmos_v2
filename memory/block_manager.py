"""
Block manager with prefix caching support.

Design Philosophy:
- Hash-based prefix caching for sharing KV cache blocks
- Reference counting for safe block reuse
- Copy-on-write semantics when blocks are shared

Optimization Notes (from benchmark report):
- Prefix caching is a key optimization in nano-vLLM
- Uses xxhash for fast, collision-resistant hashing
- Enables sharing of common prompt prefixes across requests
"""
from collections import deque
from typing import Dict, List, Set
import xxhash
import numpy as np

from memory.types import BlockManager as BlockManagerABC
from engine.sequence import Sequence


class Block:
    """
    Represents a single KV cache block.

    Attributes:
        block_id: Unique block identifier
        ref_count: Number of sequences using this block
        hash: Hash of token IDs in this block (-1 if not full)
        token_ids: Token IDs stored in this block
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: List[int] = []

    def update(self, hash: int, token_ids: List[int]) -> None:
        """Update block with new hash and token IDs."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self) -> None:
        """Reset block to initial state with ref_count=1."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager(BlockManagerABC):
    """
    Manages KV cache blocks with prefix caching.

    The block manager:
    1. Allocates blocks for new sequences
    2. Reuses blocks for shared prefixes (prefix caching)
    3. Tracks reference counts for safe sharing
    4. Deallocates blocks when no longer needed
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        Initialize block manager.

        Args:
            num_blocks: Total number of blocks available
            block_size: Number of tokens per block
        """
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1) -> int:
        """
        Compute hash for a sequence of token IDs.

        Uses xxhash for fast, high-quality hashing.
        Includes prefix hash to enable incremental hashing.

        Args:
            token_ids: Token IDs to hash
            prefix: Hash of previous block (-1 if first block)

        Returns:
            64-bit integer hash
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Mark block as allocated and return it."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} already in use"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Mark block as free and return to pool."""
        assert self.blocks[block_id].ref_count == 0, \
            f"Cannot deallocate block {block_id} with ref_count > 0"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        Check if enough free blocks exist for sequence.

        Args:
            seq: Sequence to check

        Returns:
            True if allocation is possible
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """
        Allocate blocks for sequence, reusing cached blocks if possible.

        This implements prefix caching:
        1. For each block, compute hash of its tokens
        2. Check if a block with same hash exists (cache hit)
        3. If hit and tokens match, increment ref_count and reuse
        4. If miss, allocate new block from free pool

        Args:
            seq: Sequence to allocate blocks for
        """
        assert not seq.block_table, "Sequence already has blocks allocated"

        h = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            # Only hash full blocks (partial blocks can't be cached)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # Check for cached block
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # Cache miss or hash collision
                cache_miss = True

            if cache_miss:
                # Allocate new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache hit - reuse block
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # Block already allocated, increment ref count
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block was cached but not currently used
                    block = self._allocate_block(block_id)

            # Update block hash and mapping
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        """
        Deallocate blocks used by sequence.

        Decrements reference counts. Only frees blocks when ref_count reaches 0.

        Args:
            seq: Sequence to deallocate
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        Check if we can append a token to sequence.

        Returns False if we're at block boundary and no free blocks exist.

        Args:
            seq: Sequence to check

        Returns:
            True if append is possible
        """
        # Need new block if we're at block boundary (last block is full)
        need_new_block = (len(seq) % self.block_size == 1)
        return len(self.free_block_ids) >= need_new_block

    def may_append(self, seq: Sequence) -> None:
        """
        Prepare for appending a token to sequence.

        Allocates new block if current block is full.
        Updates hash when block becomes full.

        Args:
            seq: Sequence to prepare for append
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        # Check if we need to allocate a new block
        if seq.num_blocks > len(block_table):
            # Last block is full, allocate new one
            assert last_block.hash != -1, "Last block should have hash when full"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            # Just filled last block, compute and store hash (if not already set by chunked prefill)
            if last_block.hash == -1:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id

        else:
            # Middle of block, nothing to do
            if last_block.hash != -1:
                # Hash might be set by chunked prefill for partial blocks - this is ok
                pass

    def update_hashes_after_prefill_chunk(self, seq: Sequence, chunk_start: int, chunk_end: int) -> None:
        """
        Update block hashes after processing a prefill chunk.

        During chunked prefill, we process tokens in chunks. After each chunk,
        we need to compute and store hashes for any blocks that got filled.

        Args:
            seq: Sequence that just processed a chunk
            chunk_start: Start index of the chunk (inclusive)
            chunk_end: End index of the chunk (exclusive)
        """
        block_table = seq.block_table

        # Determine which blocks were affected by this chunk
        start_block_idx = chunk_start // self.block_size
        end_block_idx = (chunk_end - 1) // self.block_size  # Inclusive

        # Update hashes for all complete blocks in this range
        for block_idx in range(start_block_idx, end_block_idx + 1):
            block = self.blocks[block_table[block_idx]]

            # Calculate how many tokens are in this block after the chunk
            block_start_token = block_idx * self.block_size
            block_end_token = min((block_idx + 1) * self.block_size, chunk_end)
            tokens_in_block = block_end_token - block_start_token + (chunk_start - block_start_token if block_idx == start_block_idx and chunk_start > block_start_token else 0)

            # If block is complete (has 256 tokens) and doesn't have a hash yet, compute it
            if block_end_token == (block_idx + 1) * self.block_size and block.hash == -1:
                token_ids = seq.block(block_idx)
                prefix = self.blocks[block_table[block_idx - 1]].hash if block_idx > 0 else -1
                h = self.compute_hash(token_ids, prefix)
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id

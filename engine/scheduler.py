"""
Scheduler for continuous batching with prefill/decode separation.

Design Philosophy:
- Maximize GPU utilization through dynamic batching
- Separate prefill (first token) and decode (subsequent tokens) phases
- Support preemption when memory is constrained
- Integrate tightly with BlockManager for memory decisions

Optimization Notes (from benchmark report):
- Continuous batching: 10-50x throughput improvement
- Key insight: GPU can process multiple sequences in parallel
- Prefill/decode separation: Different compute patterns, batch separately
- Preemption: Gracefully handle memory pressure by pausing sequences

Architecture:
- Waiting queue: Sequences waiting to be scheduled
- Running queue: Sequences currently being processed
- Scheduler decides which sequences run in each step
"""
from collections import deque
from typing import List, Tuple

from config import Config
from engine.types import Scheduler as SchedulerABC
from engine.sequence import Sequence, SequenceStatus
from memory.block_manager import BlockManager


class Scheduler(SchedulerABC):
    """
    Continuous batching scheduler with prefill/decode separation.

    The scheduler maintains two queues:
    1. Waiting: New sequences waiting to start
    2. Running: Sequences currently being processed

    In each step, the scheduler:
    - Tries to schedule waiting sequences (prefill phase)
    - If no waiting sequences, schedules running sequences (decode phase)
    - Handles preemption if memory is insufficient
    """

    def __init__(self, config: Config):
        """
        Initialize scheduler.

        Args:
            config: System configuration with batch/memory limits
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_id = config.eos_token_id

        # Set class-level configuration for Sequence
        Sequence.chunk_size = config.prefill_chunk_size

        # Block manager for KV cache memory
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size
        )

        # Sequence queues
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """
        Check if all sequences are complete.

        Returns:
            True if no sequences are waiting or running
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """
        Add a new sequence to the waiting queue.

        Args:
            seq: Sequence to add
        """
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """
        Schedule sequences for the next execution step with chunked prefill.

        Strategy (decode-maximal batching):
        1. Decode phase FIRST: Schedule all running sequences in decode phase
           - Each generates one token
           - Prioritized for low latency

        2. Prefill chunks: Fill remaining capacity with prefill chunks
           - From waiting sequences (new requests)
           - Or running sequences still processing prompt (chunked prefill)
           - Chunks limited by chunk_size (default 512 tokens)

        This mixes decode and prefill in same batch for better GPU utilization.

        Returns:
            Tuple of (sequences_to_run, is_prefill)
            - is_prefill=True if ANY sequence is doing prefill (even if mixed batch)
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        has_prefill = False

        # PHASE 1: Schedule all decode requests first (decode-maximal)
        # Iterate through running queue to find sequences ready for decode
        decode_seqs = []
        for seq in list(self.running):
            # Sequence is in decode if it has completed all prefill
            if not seq.has_remaining_prefill:
                # Check if we can append a token
                if not self.block_manager.can_append(seq):
                    continue  # Skip, will handle preemption later if needed

                # Check token budget (decode = 1 token per seq)
                if num_batched_tokens + 1 <= self.max_num_batched_tokens and num_seqs < self.max_num_seqs:
                    decode_seqs.append(seq)
                    num_batched_tokens += 1
                    num_seqs += 1

        # Reserve memory for decode tokens
        for seq in decode_seqs:
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

        # PHASE 2: Fill remaining capacity with prefill chunks
        # Try running sequences that still have prefill remaining (chunked prefill)
        for seq in list(self.running):
            if seq.has_remaining_prefill and seq not in scheduled_seqs:
                chunk_size = seq.get_next_prefill_chunk_size()

                # Check constraints
                if (num_batched_tokens + chunk_size > self.max_num_batched_tokens or
                    num_seqs >= self.max_num_seqs):
                    break

                # Schedule this prefill chunk
                scheduled_seqs.append(seq)
                num_batched_tokens += chunk_size
                num_seqs += 1
                has_prefill = True

        # Try new waiting sequences if still have capacity
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # Calculate tokens needed for first chunk
            tokens_needed = min(
                seq.num_prompt_tokens - seq.num_cached_tokens,
                seq.chunk_size
            )

            # Check constraints
            if (num_batched_tokens + tokens_needed > self.max_num_batched_tokens or
                not self.block_manager.can_allocate(seq)):
                break

            # Allocate and schedule
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append(seq)
            num_batched_tokens += tokens_needed
            num_seqs += 1
            has_prefill = True

            # Move to running queue
            self.waiting.popleft()
            self.running.append(seq)

        # Must have at least one sequence
        assert scheduled_seqs, "No sequences could be scheduled!"

        # Determine if this is a prefill or decode batch
        # We mark it as prefill if ANY sequence is doing prefill
        is_prefill = has_prefill or len(decode_seqs) == 0

        return scheduled_seqs, is_prefill

    def preempt(self, seq: Sequence) -> None:
        """
        Preempt a sequence by deallocating its memory and moving to waiting.

        The sequence will be rescheduled later when memory is available.

        Args:
            seq: Sequence to preempt
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: List[Sequence],
        token_ids: List[int],
        was_prefill: bool
    ) -> None:
        """
        Process generated tokens and update sequence states.

        With chunked prefill:
        - If sequence was in prefill: mark chunk computed, append first generated token
        - If sequence was in decode: append next generated token

        Args:
            seqs: Sequences that just executed
            token_ids: Generated token IDs (one per sequence)
            was_prefill: Whether this batch included any prefill
        """
        for seq, token_id in zip(seqs, token_ids):
            # Check if this sequence was doing prefill
            if was_prefill and seq.has_remaining_prefill:
                # Mark prefill chunk as computed
                chunk_size = seq.get_next_prefill_chunk_size()
                seq.mark_prefill_chunk_computed(chunk_size)

            # Append the generated token (first token after prefill, or next decode token)
            seq.append_token(token_id)

            # Check stopping criteria (only for decode phase)
            is_eos = (not seq.ignore_eos and token_id == self.eos_token_id)
            is_max_len = (seq.num_completion_tokens == seq.max_tokens)

            if is_eos or is_max_len:
                # Sequence is finished
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

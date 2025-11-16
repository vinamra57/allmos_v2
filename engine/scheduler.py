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
        Schedule sequences for the next execution step.

        Strategy:
        1. Prefill phase: Schedule waiting sequences (first token generation)
           - Try to schedule as many as fit within token/memory budgets
           - Each sequence processes all prompt tokens at once

        2. Decode phase: Schedule running sequences (subsequent tokens)
           - Each sequence generates one token
           - Handle preemption if memory is insufficient

        Returns:
            Tuple of (sequences_to_run, is_prefill)
        """
        # Try prefill phase first (prioritize new sequences)
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # Check constraints:
            # 1. Token budget: Don't exceed max_num_batched_tokens
            # 2. Memory budget: Check if we can allocate blocks
            tokens_needed = len(seq) - seq.num_cached_tokens

            if (num_batched_tokens + tokens_needed > self.max_num_batched_tokens or
                not self.block_manager.can_allocate(seq)):
                # Can't schedule this sequence, stop trying
                break

            # Schedule the sequence
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += tokens_needed
            seq.status = SequenceStatus.RUNNING

            # Move from waiting to running
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        # If we scheduled any prefill sequences, return them
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode phase: Schedule running sequences
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # Check if we can append a token (may need new block)
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt a sequence to free memory
                    victim = self.running.pop()
                    self.preempt(victim)
                else:
                    # No other sequences to preempt, preempt this one
                    self.preempt(seq)
                    break
            else:
                # Successfully reserved space for append
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # Must have at least one sequence to run
        assert scheduled_seqs, "No sequences could be scheduled!"

        # Put scheduled sequences back at front of running queue
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

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

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> None:
        """
        Process generated tokens and update sequence states.

        For each sequence:
        1. Append the generated token
        2. Check if generation is complete (EOS or max_tokens reached)
        3. If complete, deallocate memory and remove from running queue

        Args:
            seqs: Sequences that just generated tokens
            token_ids: Generated token IDs (one per sequence)
        """
        for seq, token_id in zip(seqs, token_ids):
            # Append the new token
            seq.append_token(token_id)

            # Check stopping criteria
            is_eos = (not seq.ignore_eos and token_id == self.eos_token_id)
            is_max_len = (seq.num_completion_tokens == seq.max_tokens)

            if is_eos or is_max_len:
                # Sequence is finished
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

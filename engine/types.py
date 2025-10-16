"""
Abstract base classes for engine components.

Design Philosophy:
- Define clear interfaces for all swappable components
- Enable modular testing and experimentation
- Document expected behavior through abstract methods
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch


class Scheduler(ABC):
    """
    Abstract scheduler for managing sequence execution.

    Responsibilities:
    - Decide which sequences to run in each step
    - Manage waiting and running queues
    - Handle preemption when memory is tight
    """

    @abstractmethod
    def add(self, seq) -> None:
        """Add a new sequence to the waiting queue."""
        pass

    @abstractmethod
    def schedule(self) -> Tuple[List, bool]:
        """
        Schedule sequences for the next step.

        Returns:
            Tuple of (sequences_to_run, is_prefill)
        """
        pass

    @abstractmethod
    def postprocess(self, seqs: List, token_ids: List[int]) -> None:
        """Process generated tokens and update sequence states."""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if all sequences are complete."""
        pass


class ModelRunner(ABC):
    """
    Abstract model runner for executing forward passes.

    Responsibilities:
    - Run model forward passes efficiently
    - Manage KV cache
    - Capture and replay CUDA graphs
    """

    @abstractmethod
    def run(self, seqs: List, is_prefill: bool) -> List[int]:
        """
        Run model on batch of sequences.

        Args:
            seqs: List of sequences to process
            is_prefill: Whether this is prefill (first token) or decode

        Returns:
            List of generated token IDs
        """
        pass

    @abstractmethod
    def warmup_model(self) -> None:
        """Warm up model to measure memory usage."""
        pass

    @abstractmethod
    def allocate_kv_cache(self) -> None:
        """Allocate KV cache based on available GPU memory."""
        pass

    @abstractmethod
    def exit(self) -> None:
        """Cleanup resources (CUDA graphs, distributed processes, etc.)."""
        pass


class LLMEngine(ABC):
    """
    Abstract LLM engine for high-level inference orchestration.

    Responsibilities:
    - Manage scheduler and model runner
    - Handle request submission
    - Execute generation loop
    """

    @abstractmethod
    def generate(self, prompts: List[str], sampling_params) -> List[dict]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters

        Returns:
            List of dictionaries with 'text' and 'token_ids' keys
        """
        pass

    @abstractmethod
    def add_request(self, prompt: str, sampling_params) -> None:
        """Add a generation request to the queue."""
        pass

    @abstractmethod
    def step(self) -> Tuple[List, int]:
        """
        Execute one step of generation.

        Returns:
            Tuple of (completed_outputs, num_tokens_processed)
        """
        pass

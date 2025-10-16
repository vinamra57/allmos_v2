"""
LLM Engine for high-level inference orchestration.

Design Philosophy:
- Simple API: generate(prompts, sampling_params)
- Coordinate scheduler and model runner
- Handle tokenization/detokenization
- Support tensor parallelism via multiprocessing
- Progress tracking with tqdm

Architecture:
- Main process (rank 0): Manages scheduler, coordinates generation
- Worker processes (rank 1-N): Run model forward passes
- Communication via multiprocessing events and shared memory
"""
import atexit
from dataclasses import fields
from time import perf_counter
from typing import List, Tuple
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import Config
from sampling_params import SamplingParams
from engine.types import LLMEngine as LLMEngineABC
from engine.sequence import Sequence
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner


class LLMEngine(LLMEngineABC):
    """
    High-level LLM inference engine.

    Orchestrates:
    1. Tokenization of prompts
    2. Scheduling of sequences (continuous batching)
    3. Model execution (via ModelRunner)
    4. Token sampling
    5. Detokenization of outputs

    Supports tensor parallelism by spawning worker processes.
    """

    def __init__(self, model: str, **kwargs):
        """
        Initialize LLM engine.

        Args:
            model: Path to model directory
            **kwargs: Configuration parameters (passed to Config)

        Example:
            >>> engine = LLMEngine(
            ...     model="~/huggingface/Qwen3-0.6B/",
            ...     tensor_parallel_size=1,
            ...     enable_cuda_graphs=True
            ... )
        """
        # Extract config parameters from kwargs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Spawn worker processes for tensor parallelism
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event)
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # Initialize main model runner (rank 0)
        self.model_runner = ModelRunner(config, 0, self.events)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=True
        )

        # Set EOS token in config
        config.eos_token_id = self.tokenizer.eos_token_id

        # Initialize scheduler
        self.scheduler = Scheduler(config)

        # Register cleanup on exit
        atexit.register(self.exit)

    def exit(self) -> None:
        """Cleanup worker processes and resources."""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | List[int],
        sampling_params: SamplingParams
    ) -> None:
        """
        Add a generation request to the scheduler.

        Args:
            prompt: Text prompt or token IDs
            sampling_params: Sampling configuration
        """
        # Tokenize if needed
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # Create sequence and add to scheduler
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> Tuple[List, int]:
        """
        Execute one step of generation.

        Steps:
        1. Scheduler decides which sequences to run
        2. Model runner executes forward pass
        3. Scheduler postprocesses results
        4. Return completed sequences

        Returns:
            Tuple of (completed_outputs, num_tokens_processed)
            - completed_outputs: List of (seq_id, token_ids) for finished sequences
            - num_tokens_processed: Positive for prefill, negative for decode
        """
        # Schedule sequences
        seqs, is_prefill = self.scheduler.schedule()

        # Run model
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # Postprocess
        self.scheduler.postprocess(seqs, token_ids)

        # Collect finished sequences
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]

        # Count tokens (positive for prefill, negative for decode)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        return outputs, num_tokens

    def is_finished(self) -> bool:
        """Check if all sequences are complete."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        use_tqdm: bool = True,
    ) -> List[dict]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of text prompts or token ID lists
            sampling_params: Sampling configuration (single or per-prompt)
            use_tqdm: Show progress bar

        Returns:
            List of dictionaries with keys:
            - 'text': Decoded completion text
            - 'token_ids': Completion token IDs

        Example:
            >>> outputs = engine.generate(
            ...     prompts=["Hello", "Introduce yourself"],
            ...     sampling_params=SamplingParams(temperature=0.8, max_tokens=100)
            ... )
            >>> print(outputs[0]['text'])
        """
        # Setup progress bar
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True
            )

        # Handle single sampling_params
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Add all requests
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # Generation loop
        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            # Update throughput metrics
            if use_tqdm:
                elapsed = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                else:
                    decode_throughput = -num_tokens / elapsed

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # Collect completed sequences
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Sort by seq_id to preserve input order
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # Decode token IDs to text
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids
            }
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return outputs

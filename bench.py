"""
Benchmark script for allmos_v2.

Matches nano-vLLM benchmark methodology:
- 256 sequences
- Random input lengths (100-1024 tokens)
- Random output lengths (100-1024 tokens)
- Temperature sampling

Target: 1434 tokens/sec (matching nano-vLLM on comparable hardware)
"""
import os
import time
from random import randint, seed

from llm import LLM
from sampling_params import SamplingParams


def main():
    # Set random seed for reproducibility
    seed(0)

    # Benchmark configuration
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    # Model path
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    print("=" * 80)
    print("Allmos v2 Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Number of sequences: {num_seqs}")
    print(f"  Input length range: 100-{max_input_len} tokens")
    print(f"  Output length range: 100-{max_output_len} tokens")
    print("=" * 80)

    # Initialize engine
    print("\nInitializing engine...")
    llm = LLM(
        model_path,
        enforce_eager=True,  # Disable CUDA graphs (flash-attn not available)
        max_model_len=4096,
        enable_prefix_caching=True,
    )

    # Generate random token sequences
    print("Generating random prompts...")
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]

    # Generate random sampling parameters
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    # Warmup (first run is slower due to compilation)
    print("\nWarming up...")
    llm.generate(["Benchmark warmup"], SamplingParams(max_tokens=10))

    # Run benchmark
    print("\nRunning benchmark...")
    print("=" * 80)

    t_start = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t_end = time.time()

    # Calculate metrics
    elapsed = t_end - t_start
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed

    # Print results
    print("=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(f"Total output tokens: {total_tokens:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print("=" * 80)

    # Compare to baselines
    print("\nComparison to Baselines:")
    print(f"  Original Allmos:  22.81 tokens/sec (baseline)")
    print(f"  nano-vLLM:        1434.13 tokens/sec (target)")
    print(f"  allmos_v2:        {throughput:.2f} tokens/sec")
    print(f"\nSpeedup vs original: {throughput / 22.81:.1f}x")
    print(f"vs nano-vLLM target: {throughput / 1434.13:.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()

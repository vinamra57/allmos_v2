"""
Benchmark script with FP8 quantization enabled.
"""
import os
import time
from random import randint, seed
from llm import LLM
from sampling_params import SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # Enable FP8 quantization
    llm = LLM(path, enforce_eager=False, max_model_len=4096, enable_fp8=True)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # Actual benchmark
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"\n{'='*80}")
    print(f"FP8 BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {t:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"Baseline (FP16): 1,758 tok/s")
    print(f"Speedup: {throughput/1758:.2f}x")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

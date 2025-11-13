import os
import sys
import time
from random import randint, seed
from llm import LLM
from sampling_params import SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    print("="*50, flush=True)
    print("ALLMOS_V2 BENCHMARK STARTING", flush=True)
    print("="*50, flush=True)

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    print(f"Loading model from: {path}", flush=True)
    llm = LLM(path, enforce_eager=True, max_model_len=4096)  # Disable CUDA graphs temporarily
    print("Model loaded successfully!", flush=True)

    print(f"Preparing {num_seqs} sequences...", flush=True)
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"Total tokens to generate: {total_tokens}", flush=True)

    print("\nRunning warmup...", flush=True)
    llm.generate(["Benchmark: "], SamplingParams(max_tokens=10))
    print("Warmup complete!\n", flush=True)

    print("Starting main benchmark...", flush=True)
    sys.stdout.flush()
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    throughput = total_tokens / elapsed
    print(f"\n{'='*50}", flush=True)
    print(f"BENCHMARK RESULTS:", flush=True)
    print(f"Total tokens: {total_tokens}", flush=True)
    print(f"Time: {elapsed:.2f}s", flush=True)
    print(f"Throughput: {throughput:.2f} tok/s", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()

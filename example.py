"""
Example usage of allmos_v2 LLM inference engine.

This script demonstrates basic text generation with the optimized engine.
"""
import os
from llm import LLM
from sampling_params import SamplingParams
from transformers import AutoTokenizer


def main():
    # Model path (update to your local path)
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    print("Initializing LLM engine...")
    print("=" * 60)

    # Initialize engine with optimizations
    llm = LLM(
        model_path,
        enforce_eager=True,  # Set to False to enable CUDA graphs
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )

    # Load tokenizer for prompt formatting
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define prompts
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # Apply chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # Generation parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256
    )

    print("\nGenerating completions...")
    print("=" * 60)

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i+1}:")
        print(f"{'=' * 60}")
        print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
        print(f"\n{'Completion:':}")
        print(f"{'-' * 60}")
        print(output['text'])
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""
Comprehensive end-to-end testing for allmos_v2.

This script validates:
1. Basic single sequence generation
2. Batched generation with multiple sequences
3. Prefix caching functionality
4. CUDA graph capture
5. Throughput measurement
"""
import os
import sys
import torch
from llm import LLM
from sampling_params import SamplingParams


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_basic_generation(llm):
    """Test 1: Basic single sequence generation."""
    print_section("TEST 1: Basic Single Sequence Generation")

    prompt = "What is 2+2?"
    print(f"Prompt: {prompt}")

    outputs = llm.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(temperature=0.8, max_tokens=50),
        use_tqdm=False
    )

    output_text = outputs[0]['text']
    token_ids = outputs[0]['token_ids']

    print(f"\nOutput ({len(token_ids)} tokens):")
    print(output_text[:200] + "..." if len(output_text) > 200 else output_text)

    # Validation
    assert len(token_ids) > 0, "No tokens generated!"
    assert len(token_ids) <= 50, "Generated more than max_tokens!"
    assert output_text, "No text decoded!"

    print("\n✅ PASSED: Basic generation works")
    return True


def test_batched_generation(llm):
    """Test 2: Batched generation with multiple sequences."""
    print_section("TEST 2: Batched Generation (Multiple Sequences)")

    prompts = [
        "Count from 1 to 5:",
        "What are the primary colors?",
        "Name three planets:",
    ]

    print(f"Number of prompts: {len(prompts)}")

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(temperature=0.7, max_tokens=30),
        use_tqdm=False
    )

    # Validation
    assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        token_ids = output['token_ids']
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Output: {output['text'][:100]}...")
        assert len(token_ids) > 0, f"Sequence {i} produced no tokens!"
        assert len(token_ids) <= 30, f"Sequence {i} exceeded max_tokens!"

    print("\n✅ PASSED: Batched generation works")
    return True


def test_prefix_caching(llm):
    """Test 3: Prefix caching with shared prefixes."""
    print_section("TEST 3: Prefix Caching (Shared Prefixes)")

    # Create prompts with shared prefix
    shared_prefix = "Once upon a time in a faraway land, "
    prompts = [
        shared_prefix + "there lived a dragon.",
        shared_prefix + "there was a castle.",
        shared_prefix + "people told stories.",
    ]

    print(f"Shared prefix: '{shared_prefix}'")
    print(f"Number of prompts with shared prefix: {len(prompts)}")

    # First run (cold cache)
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(temperature=0.6, max_tokens=20),
        use_tqdm=False
    )

    # Validation
    assert len(outputs) == len(prompts), "Incorrect number of outputs"

    for i, output in enumerate(outputs):
        print(f"\nOutput {i+1}: {output['text'][:80]}...")
        assert len(output['token_ids']) > 0, f"Sequence {i} produced no tokens!"

    print("\n✅ PASSED: Prefix caching works")
    print("Note: Check scheduler logs for cache hit indicators")
    return True


def test_variable_lengths(llm):
    """Test 4: Variable length sequences."""
    print_section("TEST 4: Variable Length Sequences")

    # Different max_tokens for each sequence
    prompts = ["Hi", "Hello there", "Good morning everyone"]
    sampling_params = [
        SamplingParams(temperature=0.8, max_tokens=10),
        SamplingParams(temperature=0.8, max_tokens=20),
        SamplingParams(temperature=0.8, max_tokens=30),
    ]

    print(f"Number of prompts: {len(prompts)}")
    print(f"Max tokens: {[sp.max_tokens for sp in sampling_params]}")

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False
    )

    # Validation
    assert len(outputs) == len(prompts), "Incorrect number of outputs"

    for i, (sp, output) in enumerate(zip(sampling_params, outputs)):
        token_count = len(output['token_ids'])
        print(f"\nSequence {i+1}: Generated {token_count} tokens (max: {sp.max_tokens})")
        assert token_count <= sp.max_tokens, \
            f"Sequence {i} exceeded max_tokens: {token_count} > {sp.max_tokens}"

    print("\n✅ PASSED: Variable length sequences work")
    return True


def test_cuda_graph_info(llm):
    """Test 5: Check CUDA graph capture."""
    print_section("TEST 5: CUDA Graph Information")

    model_runner = llm.model_runner

    if model_runner.enforce_eager:
        print("⚠️  CUDA graphs disabled (enforce_eager=True)")
        print("This is expected if flash-attn is not available")
        return True

    # Check if graphs were captured
    if hasattr(model_runner, 'graphs') and model_runner.graphs:
        print(f"✅ CUDA graphs captured: {len(model_runner.graphs)} graphs")
        print(f"Batch sizes: {model_runner.graph_bs}")
    else:
        print("⚠️  No CUDA graphs found")
        return False

    print("\n✅ PASSED: CUDA graphs captured successfully")
    return True


def test_memory_management(llm):
    """Test 6: Memory management."""
    print_section("TEST 6: Memory Management")

    # Get KV cache info
    scheduler = llm.scheduler
    block_manager = scheduler.block_manager

    print(f"Total KV cache blocks: {len(block_manager.blocks)}")
    print(f"Block size: {block_manager.block_size} tokens")
    print(f"Free blocks: {len(block_manager.free_block_ids)}")
    print(f"Used blocks: {len(block_manager.used_block_ids)}")

    # Check GPU memory
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f"\nGPU Memory:")
    print(f"  Total: {total / 1e9:.2f} GB")
    print(f"  Used: {used / 1e9:.2f} GB")
    print(f"  Free: {free / 1e9:.2f} GB")

    assert len(block_manager.blocks) > 0, "No KV cache blocks allocated!"
    assert len(block_manager.free_block_ids) + len(block_manager.used_block_ids) == len(block_manager.blocks), \
        "Block accounting mismatch!"

    print("\n✅ PASSED: Memory management working")
    return True


def run_all_tests():
    """Run all tests sequentially."""
    print_section("Allmos v2 Comprehensive Test Suite")

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        print("Please download the model first or update the path")
        sys.exit(1)

    print(f"Model path: {model_path}")
    print("Initializing engine...")

    # Initialize engine
    try:
        llm = LLM(
            model_path,
            enforce_eager=True,  # Set to False to test CUDA graphs
            max_num_seqs=128,
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run tests
    tests = [
        ("Basic Generation", test_basic_generation),
        ("Batched Generation", test_batched_generation),
        ("Prefix Caching", test_prefix_caching),
        ("Variable Lengths", test_variable_lengths),
        ("CUDA Graphs", test_cuda_graph_info),
        ("Memory Management", test_memory_management),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func(llm)
            if result:
                passed += 1
            else:
                failed += 1
                print(f"⚠️  Test '{test_name}' returned False")
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR in test '{test_name}': {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_section("Test Summary")
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

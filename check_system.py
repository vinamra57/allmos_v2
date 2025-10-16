"""
System check script for allmos_v2.

This script verifies:
1. Python version compatibility
2. Required dependencies are installed
3. Optional dependencies availability
4. CUDA availability and version
5. GPU memory
6. Model files exist
7. All modules can be imported

Run this BEFORE attempting to use allmos_v2.
"""
import sys
import os


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_python_version():
    """Check Python version >= 3.10."""
    print_header("Python Version Check")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ FAIL: Python 3.10+ required")
        print("   Please upgrade Python")
        return False

    print("✅ PASS: Python version compatible")
    return True


def check_required_packages():
    """Check all required packages are installed."""
    print_header("Required Package Check")

    required = {
        "torch": "2.4.0",
        "transformers": "4.51.0",
        "xxhash": "3.0.0",
        "safetensors": "0.4.0",
        "tqdm": "4.65.0",
        "numpy": "1.26.0",
    }

    all_ok = True
    for package, min_version in required.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED (required >= {min_version})")
            all_ok = False

    if not all_ok:
        print("\n❌ FAIL: Missing required packages")
        print("   Run: pip install -r requirements.txt")
        return False

    print("\n✅ PASS: All required packages installed")
    return True


def check_optional_packages():
    """Check optional optimization packages."""
    print_header("Optional Package Check")

    # Check flash_attn
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "unknown")
        print(f"✅ flash-attn: {version} (INSTALLED)")
        print("   → Flash Attention enabled (1.5-2x speedup)")
    except ImportError as e:
        print(f"⚠️  flash-attn: NOT AVAILABLE")
        print(f"   → Reason: {str(e)}")
        print("   → Will use PyTorch fallback (slower but functional)")
        print("   → To fix: Check GLIBC version (need 2.32+)")

    # Check triton
    try:
        import triton
        version = getattr(triton, "__version__", "unknown")
        print(f"\n✅ triton: {version} (INSTALLED)")
        print("   → Triton kernels enabled")
    except ImportError as e:
        print(f"\n⚠️  triton: NOT AVAILABLE")
        print(f"   → Reason: {str(e)}")
        print("   → Will use PyTorch fallback for KV cache storage")

    print("\n✅ INFO: Optional packages checked")
    return True


def check_cuda():
    """Check CUDA availability and version."""
    print_header("CUDA Check")

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ FAIL: CUDA not available")
            print("   PyTorch was not compiled with CUDA support")
            return False

        cuda_version = torch.version.cuda
        print(f"✅ CUDA available: version {cuda_version}")

        # Check GPU info
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPUs found: {gpu_count}")

        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"\n   GPU {i}: {name}")
            print(f"   - Memory: {memory_gb:.2f} GB")
            print(f"   - Compute Capability: {props.major}.{props.minor}")

        # Check cuDNN
        if torch.backends.cudnn.is_available():
            print(f"\n✅ cuDNN available: version {torch.backends.cudnn.version()}")
        else:
            print("\n⚠️  cuDNN not available")

        print("\n✅ PASS: CUDA environment ready")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error checking CUDA: {e}")
        return False


def check_model_files():
    """Check if model files exist."""
    print_header("Model Files Check")

    default_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    if not os.path.exists(default_path):
        print(f"⚠️  Default model path not found: {default_path}")
        print("   Please download the model or update the path in scripts")
        print("\n   To download:")
        print("   pip install huggingface-hub")
        print("   huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/")
        return False

    # Check for required files
    required_files = ["config.json", "tokenizer.json"]
    safetensors = []

    for file in required_files:
        path = os.path.join(default_path, file)
        if os.path.exists(path):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            return False

    # Check for safetensors files
    import glob
    safetensors = glob.glob(os.path.join(default_path, "*.safetensors"))
    if safetensors:
        print(f"✅ Found {len(safetensors)} safetensors file(s)")
    else:
        print(f"❌ No safetensors files found")
        return False

    print(f"\n✅ PASS: Model files ready at {default_path}")
    return True


def check_imports():
    """Check all allmos_v2 modules can be imported."""
    print_header("Module Import Check")

    modules_to_test = [
        "config",
        "sampling_params",
        "llm",
        "engine.types",
        "engine.sequence",
        "engine.scheduler",
        "engine.llm_engine",
        "memory.types",
        "memory.block_manager",
        "layers.attention",
        "layers.sampler",
        "layers.layernorm",
        "layers.activation",
        "layers.rotary_embedding",
        "layers.linear",
        "layers.embed_head",
        "models.qwen3",
        "utils.context",
        "utils.loader",
    ]

    all_ok = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {str(e)}")
            all_ok = False

    if not all_ok:
        print("\n❌ FAIL: Some modules failed to import")
        return False

    print("\n✅ PASS: All modules import successfully")
    return True


def check_glibc():
    """Check GLIBC version (Linux only)."""
    print_header("GLIBC Version Check (Linux)")

    if sys.platform != "linux":
        print(f"ℹ️  Not on Linux (platform: {sys.platform}), skipping")
        return True

    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        gnu_get_libc_version = libc.gnu_get_libc_version
        gnu_get_libc_version.restype = ctypes.c_char_p
        version = gnu_get_libc_version().decode()

        print(f"GLIBC version: {version}")

        major, minor = map(int, version.split('.')[:2])
        if major > 2 or (major == 2 and minor >= 32):
            print(f"✅ GLIBC {version} >= 2.32 (flash-attn compatible)")
        else:
            print(f"⚠️  GLIBC {version} < 2.32 (flash-attn NOT compatible)")
            print("   → Will use PyTorch fallback (enforce_eager=True recommended)")
            print("   → Or upgrade to Ubuntu 22.04+/Debian 12+")

    except Exception as e:
        print(f"⚠️  Could not determine GLIBC version: {e}")

    return True


def main():
    """Run all checks."""
    print("=" * 80)
    print("  allmos_v2 System Check")
    print("=" * 80)
    print("\nThis script verifies your system is ready to run allmos_v2")

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Optional Packages", check_optional_packages),
        ("CUDA Environment", check_cuda),
        ("Model Files", check_model_files),
        ("Module Imports", check_imports),
        ("GLIBC Version", check_glibc),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print_header("Summary")

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL CHECKS PASSED! System is ready to use allmos_v2")
        print("\nNext steps:")
        print("  1. Run example: python example.py")
        print("  2. Run tests: python test_allmos.py")
        print("  3. Run benchmark: python bench.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please address issues above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing model: huggingface-cli download Qwen/Qwen3-0.6B")
        print("  - GLIBC < 2.32: Set enforce_eager=True in LLM initialization")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

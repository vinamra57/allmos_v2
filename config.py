"""
Centralized configuration for allmos_v2.

Design Philosophy:
- Type-safe configuration using dataclasses
- Auto-computed values for memory management
- Support for all optimization features (CUDA graphs, prefix caching, etc.)
"""
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """Main configuration for the LLM inference engine."""

    # Model configuration
    model: str
    max_model_len: int = 4096

    # Batching configuration
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384

    # Memory configuration
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1  # Auto-computed

    # Optimization flags
    enable_cuda_graphs: bool = True
    enable_prefix_caching: bool = True
    enforce_eager: bool = False

    # Parallelism
    tensor_parallel_size: int = 1

    # Auto-loaded configurations
    hf_config: AutoConfig = None
    eos_token_id: int = -1

    def __post_init__(self):
        """Validate and auto-compute configuration values."""
        assert os.path.isdir(self.model), f"Model path {self.model} does not exist"
        assert self.kvcache_block_size % 256 == 0, "Block size must be multiple of 256"
        assert 1 <= self.tensor_parallel_size <= 8, "Tensor parallel size must be 1-8"

        # Load HuggingFace config
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # Adjust max_model_len based on model's maximum
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings
        )

        # Ensure max_num_batched_tokens >= max_model_len
        assert self.max_num_batched_tokens >= self.max_model_len, \
            "max_num_batched_tokens must be >= max_model_len"

    def compute_num_kvcache_blocks(self, total_memory: int, used_memory: int,
                                   peak_memory: int, current_memory: int) -> int:
        """
        Compute number of KV cache blocks based on available GPU memory.

        Args:
            total_memory: Total GPU memory in bytes
            used_memory: Currently used memory
            peak_memory: Peak memory during warmup
            current_memory: Current allocated memory

        Returns:
            Number of KV cache blocks that can fit in memory
        """
        num_kv_heads = self.hf_config.num_key_value_heads // self.tensor_parallel_size
        head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)

        # Calculate bytes per block (for both K and V cache)
        # 2 (K and V) × num_layers × block_size × num_kv_heads × head_dim × dtype_size
        dtype_size = self.hf_config.torch_dtype.itemsize
        block_bytes = (
            2 *
            self.hf_config.num_hidden_layers *
            self.kvcache_block_size *
            num_kv_heads *
            head_dim *
            dtype_size
        )

        # Available memory for KV cache
        available = int(
            total_memory * self.gpu_memory_utilization -
            used_memory - peak_memory + current_memory
        )

        num_blocks = available // block_bytes
        assert num_blocks > 0, f"Not enough memory for KV cache. Available: {available}, needed per block: {block_bytes}"

        return num_blocks

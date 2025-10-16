"""
Attention layer with KV cache management.

Design Philosophy:
- Use flash_attn library for memory-efficient attention (if available)
- Fallback to standard PyTorch attention if flash_attn unavailable
- Support both prefill and decode phases
- Efficient KV cache storage using Triton kernels (if available)

Optimization Notes (from benchmark report):
- Flash Attention provides 1.5-2x speedup over standard attention
- O(N) memory vs O(N²) for standard attention
- Fused kernels reduce memory bandwidth usage

IMPORTANT: flash-attn requires GLIBC 2.32+ and may not be available on all systems.
If unavailable, we gracefully fall back to PyTorch's scaled_dot_product_attention.
"""
import torch
from torch import nn
import torch.nn.functional as F

# Try to import flash_attn and triton (optional dependencies)
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, using standard PyTorch attention")

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: triton not available, using PyTorch fallback for KV cache")

from utils.context import get_context


# Triton kernel for KV cache storage (only if triton available)
if TRITON_AVAILABLE:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """
        Triton kernel for efficiently storing K/V states into cache.

        This fused kernel stores keys and values directly to their cache locations
        in a single pass, avoiding intermediate copies.
        """
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)

        # Skip if no slot assigned (padding)
        if slot == -1:
            return

        # Load key and value
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)

        # Store to cache
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
) -> None:
    """
    Store key/value states into KV cache.

    Args:
        key: [N, num_kv_heads, head_dim] key states
        value: [N, num_kv_heads, head_dim] value states
        k_cache: [num_blocks, block_size, num_kv_heads * head_dim] key cache
        v_cache: [num_blocks, block_size, num_kv_heads * head_dim] value cache
        slot_mapping: [N] mapping from token index to cache slot
    """
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim

    # Note: Triton kernel disabled due to stride compatibility issues with flattened cache
    # The PyTorch fallback is fast enough for KV cache storage
    if False and TRITON_AVAILABLE:
        # Fast path: use Triton kernel
        assert key.stride(-1) == 1 and value.stride(-1) == 1, "Last dim must be contiguous"
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N

        store_kvcache_kernel[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache, v_cache,
            slot_mapping,
            D
        )
    else:
        # Fallback: use PyTorch indexing
        key = key.reshape(N, D)
        value = value.reshape(N, D)

        for i in range(N):
            slot = slot_mapping[i].item()
            if slot == -1:
                continue
            block_idx = slot // k_cache.size(1)
            slot_idx = slot % k_cache.size(1)
            k_cache[block_idx, slot_idx] = key[i]
            v_cache[block_idx, slot_idx] = value[i]


class Attention(nn.Module):
    """
    Flash Attention with KV cache support.

    Handles both prefill (first token) and decode (subsequent tokens) phases:
    - Prefill: Use flash_attn_varlen_func for variable-length batching
    - Decode: Use flash_attn_with_kvcache for efficient KV cache reuse
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        """
        Initialize attention layer.

        Args:
            num_heads: Number of query heads
            head_dim: Dimension of each head
            scale: Attention scaling factor (typically 1/sqrt(head_dim))
            num_kv_heads: Number of key/value heads (for GQA)
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # KV cache tensors (will be set by model_runner)
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention output.

        Args:
            q: Query states [N, num_heads, head_dim]
            k: Key states [N, num_kv_heads, head_dim]
            v: Value states [N, num_kv_heads, head_dim]

        Returns:
            Attention output [N, num_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Debug
        if context.is_prefill:
            print(f"[ATT DEBUG] Input - q: {q.shape}, k: {k.shape}, v: {v.shape}")

            # Store K/V into cache if cache is allocated
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if FLASH_ATTN_AVAILABLE:
            # Fast path: use Flash Attention
            if context.is_prefill:
                # Prefill phase: compute attention for all prompt tokens
                if context.block_tables is not None:
                    # Use cached K/V for prefix caching
                    k, v = k_cache, v_cache

                output = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables
                )
            else:
                # Decode phase: compute attention for single token using KV cache
                output = flash_attn_with_kvcache(
                    q.unsqueeze(1),  # Add sequence dimension
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True
                )
        else:
            # Fallback: use standard PyTorch attention
            output = self._standard_attention(q, k, v, context, k_cache, v_cache)

        # Debug
        if context.is_prefill:
            print(f"[ATT DEBUG] Output - output: {output.shape}")

        return output

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Repeat K/V heads to match the number of Q heads (for GQA).

        Args:
            hidden_states: [batch_size, seq_len, num_kv_heads, head_dim]

        Returns:
            Repeated tensor: [batch_size, seq_len, num_heads, head_dim]
        """
        if self.num_kv_heads == self.num_heads:
            return hidden_states

        n_rep = self.num_heads // self.num_kv_heads
        # Repeat K/V heads to match Q heads
        # [..., num_kv_heads, head_dim] -> [..., num_heads, head_dim]
        return hidden_states.repeat_interleave(n_rep, dim=-2)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor
    ) -> torch.Tensor:
        """
        Fallback attention using PyTorch's scaled_dot_product_attention.

        This is used when flash_attn is not available (GLIBC < 2.32).
        Handles Grouped Query Attention (GQA) by repeating K/V heads.
        """
        # Repeat K/V heads if using GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if context.is_prefill:
            # Prefill: process all tokens at once
            # Need to handle variable-length sequences and causal masking

            # If using cached K/V (prefix caching), fetch from cache
            if context.block_tables is not None:
                # Reconstruct K/V from cache using block tables
                batch_size = context.cu_seqlens_q.size(0) - 1
                max_seqlen = context.max_seqlen_k

                # Simple fallback: process each sequence separately
                outputs = []
                for i in range(batch_size):
                    start_q = context.cu_seqlens_q[i].item()
                    end_q = context.cu_seqlens_q[i + 1].item()
                    seqlen = end_q - start_q

                    q_seq = q[start_q:end_q]  # [seqlen, num_heads, head_dim]

                    # Get K/V from cache for this sequence
                    # Cache is stored as [num_blocks, block_size, num_kv_heads * head_dim]
                    # Reshape to [num_blocks, block_size, num_kv_heads, head_dim]
                    block_table = context.block_tables[i]
                    k_seq_list = []
                    v_seq_list = []
                    for block_idx in block_table:
                        if block_idx == -1:
                            break
                        # Reshape from flattened to separate heads
                        k_block = k_cache[block_idx].view(-1, self.num_kv_heads, self.head_dim)
                        v_block = v_cache[block_idx].view(-1, self.num_kv_heads, self.head_dim)
                        k_seq_list.append(k_block)
                        v_seq_list.append(v_block)

                    k_seq = torch.cat(k_seq_list, dim=0)[:context.max_seqlen_k]
                    v_seq = torch.cat(v_seq_list, dim=0)[:context.max_seqlen_k]

                    # Repeat K/V for GQA
                    k_seq = self._repeat_kv(k_seq)
                    v_seq = self._repeat_kv(v_seq)

                    # Reshape for attention: [1, num_heads, seqlen, head_dim]
                    q_seq = q_seq.unsqueeze(0).transpose(1, 2)
                    k_seq = k_seq.unsqueeze(0).transpose(1, 2)
                    v_seq = v_seq.unsqueeze(0).transpose(1, 2)

                    # Causal attention
                    out = F.scaled_dot_product_attention(
                        q_seq, k_seq, v_seq,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True,
                        scale=self.scale
                    )

                    outputs.append(out.squeeze(0).transpose(0, 1))

                output = torch.cat(outputs, dim=0)
            else:
                # Standard prefill without cached K/V
                # Process in a single batch (assumes same length or padded)
                q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, N, head_dim]
                k = k.unsqueeze(0).transpose(1, 2)
                v = v.unsqueeze(0).transpose(1, 2)

                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=self.scale
                )
                output = output.squeeze(0).transpose(0, 1)  # [N, num_heads, head_dim]
        else:
            # Decode: single token per sequence, use KV cache
            batch_size = q.size(0)

            outputs = []
            for i in range(batch_size):
                # Get K/V from cache for this sequence
                # Cache is stored as [num_blocks, block_size, num_kv_heads * head_dim]
                # Reshape to [num_blocks, block_size, num_kv_heads, head_dim]
                block_table = context.block_tables[i]
                seqlen = context.context_lens[i].item()

                k_seq_list = []
                v_seq_list = []
                for block_idx in block_table:
                    if block_idx == -1:
                        break
                    # Reshape from flattened to separate heads
                    k_block = k_cache[block_idx].view(-1, self.num_kv_heads, self.head_dim)
                    v_block = v_cache[block_idx].view(-1, self.num_kv_heads, self.head_dim)
                    k_seq_list.append(k_block)
                    v_seq_list.append(v_block)

                k_seq = torch.cat(k_seq_list, dim=0)[:seqlen]
                v_seq = torch.cat(v_seq_list, dim=0)[:seqlen]

                # Repeat K/V for GQA
                k_seq = self._repeat_kv(k_seq)
                v_seq = self._repeat_kv(v_seq)

                # Single query token
                q_seq = q[i:i+1]  # [1, num_heads, head_dim]

                # Reshape: [1, num_heads, 1, head_dim] for query
                #          [1, num_heads, seqlen, head_dim] for key/value
                q_seq = q_seq.unsqueeze(0).transpose(1, 2).unsqueeze(2)
                k_seq = k_seq.unsqueeze(0).transpose(1, 2)
                v_seq = v_seq.unsqueeze(0).transpose(1, 2)

                out = F.scaled_dot_product_attention(
                    q_seq, k_seq, v_seq,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,  # Not needed for decode (single token)
                    scale=self.scale
                )

                outputs.append(out.squeeze(0).squeeze(1))

            output = torch.stack(outputs, dim=0)  # [batch_size, num_heads, head_dim]

        return output

"""
Qwen3 model implementation with all optimizations.

Design Philosophy:
- Use optimized layers (Flash Attention, fused ops, etc.)
- Support tensor parallelism for multi-GPU
- Fused residual connections to reduce memory bandwidth

Architecture:
- Decoder-only transformer
- Grouped Query Attention (GQA)
- SwiGLU activation in FFN
- RMSNorm instead of LayerNorm
"""
from __future__ import annotations

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen2Config as Qwen3Config

from layers.activation import SiluAndMul
from layers.attention import Attention
from layers.layernorm import RMSNorm
from layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from layers.rotary_embedding import get_rope
from layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA).

    GQA reduces KV cache size by sharing K/V across query head groups.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV projection (merged for efficiency)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # Rotary embedding
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Flash Attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # Q/K normalization (for numerical stability in Qwen3)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention forward pass.

        Args:
            positions: Token position indices
            hidden_states: Input hidden states

        Returns:
            Attention output
        """
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape to [N, num_heads, head_dim]
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v)

        # Output projection
        output = self.o_proj(attn_output.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation.

    SwiGLU: FFN_SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up)W_down
    where ⊙ is element-wise multiplication.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        assert hidden_act == "silu", f"Only SiLU activation supported, got {hidden_act}"

        # Merged gate + up projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate and up have same size
            bias=False,
        )

        # Down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        # Fused SiLU + multiply activation
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: Qwen3Config):
        super().__init__()

        # Self-attention
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # Feed-forward network
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder layer forward pass.

        Uses fused residual connections to reduce memory bandwidth.

        Args:
            positions: Token positions
            hidden_states: Input hidden states
            residual: Residual from previous layer (None for first layer)

        Returns:
            Tuple of (output hidden states, residual)
        """
        # Self-attention block
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # FFN block
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 transformer model (without LM head)."""

    def __init__(self, config: Qwen3Config):
        super().__init__()

        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Model forward pass.

        Args:
            input_ids: Input token IDs
            positions: Token positions

        Returns:
            Hidden states
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # Pass through all transformer layers
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Final normalization
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 model with language modeling head.

    This is the top-level model used for text generation.
    """

    # Mapping for loading weights from HuggingFace checkpoints
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # Tie embeddings if specified in config
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through model (returns hidden states, not logits).

        Logits are computed separately in compute_logits() for efficiency.
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Separated from forward() to support CUDA graph caching.
        """
        return self.lm_head(hidden_states)

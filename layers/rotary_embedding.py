"""
Rotary Position Embedding (RoPE).

Design Philosophy:
- Precompute cos/sin tables for efficiency
- Use torch.compile for kernel fusion
- Cache computation across different model instantiations
"""
from __future__ import annotations

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embedding to input tensor.

    Args:
        x: Input tensor [..., head_dim]
        cos: Cosine values
        sin: Sine values

    Returns:
        Rotary embedded tensor
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        """
        Initialize RoPE.

        Args:
            head_size: Dimension of each attention head
            rotary_dim: Dimension to apply rotation (usually == head_size)
            max_position_embeddings: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "Partial rotary embedding not yet supported"

        # Compute frequency bases
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # Precompute cos/sin for all positions
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # Concatenate and add batch dimension
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key.

        Args:
            positions: Position indices for each token
            query: Query tensor
            key: Key tensor

        Returns:
            Tuple of (rotated query, rotated key)
        """
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    Get or create a cached RoPE instance.

    Args:
        head_size: Dimension of each attention head
        rotary_dim: Dimension to apply rotation
        max_position: Maximum position
        base: Frequency base
        rope_scaling: Optional scaling configuration

    Returns:
        RotaryEmbedding instance
    """
    assert rope_scaling is None, "RoPE scaling not yet supported"
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)

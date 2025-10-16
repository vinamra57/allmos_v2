"""
Global context manager for attention computations.

Design Philosophy:
- Thread-local storage for attention metadata
- Avoid passing many arguments through function calls
- Support both prefill and decode phases with different metadata

Optimization Note:
This pattern is used by nano-vLLM to efficiently pass attention context
(sequence lengths, block tables, etc.) to Flash Attention without
modifying every function signature.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class AttentionContext:
    """Context information for attention computation."""

    # Common fields
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None  # Maps tokens to KV cache slots
    block_tables: torch.Tensor | None = None  # Block tables for sequences

    # Prefill-specific fields
    cu_seqlens_q: torch.Tensor | None = None  # Cumulative sequence lengths (queries)
    cu_seqlens_k: torch.Tensor | None = None  # Cumulative sequence lengths (keys)
    max_seqlen_q: int = 0                     # Maximum query sequence length
    max_seqlen_k: int = 0                     # Maximum key sequence length

    # Decode-specific fields
    context_lens: torch.Tensor | None = None  # Context length for each sequence


# Global context (thread-local in production)
_CONTEXT = AttentionContext()


def get_context() -> AttentionContext:
    """Get current attention context."""
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
) -> None:
    """Set attention context for current operation."""
    global _CONTEXT
    _CONTEXT = AttentionContext(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context() -> None:
    """Reset context to default state."""
    global _CONTEXT
    _CONTEXT = AttentionContext()

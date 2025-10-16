"""
Sampling parameters for text generation.

Design Philosophy:
- Simple, typed configuration for generation
- Validation to prevent common errors
"""
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Parameters controlling text generation sampling."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """Validate sampling parameters."""
        assert self.temperature > 1e-10, \
            "Temperature must be > 0. Greedy sampling not yet supported."
        assert self.max_tokens > 0, "max_tokens must be positive"

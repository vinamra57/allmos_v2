"""
Activation functions for transformer models.

Design Philosophy:
- Fused activations for efficiency
- Use torch.compile for kernel fusion
"""
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    Fused SiLU activation with multiplication.

    Used in transformer FFN: SiLU(gate) * up
    Fusion reduces memory bandwidth by avoiding intermediate storage.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fused SiLU and multiply.

        Args:
            x: Input tensor (concatenated gate and up projections)

        Returns:
            SiLU(gate) * up
        """
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

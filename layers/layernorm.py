"""
RMSNorm with fused residual addition.

Design Philosophy:
- Fuse residual addition with normalization to reduce memory bandwidth
- Use torch.compile for automatic kernel fusion
- Support both standalone and fused modes

Optimization Notes (from benchmark report):
- Kernel fusion provides ~1.3x speedup
- Reduces number of GPU kernel launches
- Saves memory bandwidth by avoiding intermediate tensors
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is simpler than LayerNorm (no mean subtraction or bias)
    and performs comparably for transformers.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        """
        Initialize RMSNorm.

        Args:
            hidden_size: Size of hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard RMSNorm forward pass.

        Formula: x * rsqrt(mean(x^2) + eps) * weight
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused residual addition + RMSNorm.

        This fusion reduces memory bandwidth:
        - Single kernel instead of separate add + norm
        - Avoids storing intermediate result

        Returns:
            Tuple of (normalized_output, new_residual)
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional residual fusion.

        Args:
            x: Input tensor
            residual: Optional residual tensor to add before normalization

        Returns:
            If residual is None: normalized output
            If residual is provided: (normalized output, new residual)
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

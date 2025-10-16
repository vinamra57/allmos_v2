"""
GPU-based sampling for token generation.

Design Philosophy:
- Keep sampling on GPU to avoid CPU-GPU transfers
- Use torch.compile for kernel fusion
- Implement temperature sampling efficiently

Optimization Notes (from benchmark report):
- Moving sampling to GPU provides ~1.5x speedup
- Avoids expensive argmax in Python
- Fused operations reduce memory bandwidth
"""
import torch
from torch import nn


class Sampler(nn.Module):
    """
    Efficient GPU-based token sampler.

    Uses temperature-scaled softmax sampling with Gumbel noise.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample tokens from logits using temperature scaling.

        Args:
            logits: [batch_size, vocab_size] logits
            temperatures: [batch_size] temperature values

        Returns:
            [batch_size] sampled token IDs

        Note: Uses Gumbel-max trick for efficient sampling:
            sample = argmax(logits / temp + Gumbel(0, 1))
        This is equivalent to softmax sampling but faster.
        """
        # Scale logits by temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample using Gumbel-max trick
        # Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
        gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sample_tokens = probs.div_(gumbel_noise).argmax(dim=-1)

        return sample_tokens

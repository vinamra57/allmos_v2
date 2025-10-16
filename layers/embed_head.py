"""
Embedding and language model head with tensor parallelism.

Design Philosophy:
- Vocabulary parallelism for multi-GPU setups
- Efficient gather/scatter operations for distributed embeddings
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary sharding across GPUs.

    Each GPU owns a portion of the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1

        assert num_embeddings % self.tp_size == 0, \
            f"Vocab size {num_embeddings} not divisible by tp_size {self.tp_size}"

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load sharded embedding weights."""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input tokens.

        For distributed case:
        1. Mask out tokens not owned by this GPU
        2. Lookup embeddings (zeros for non-owned tokens)
        3. All-reduce to sum contributions from all GPUs
        """
        if self.tp_size > 1:
            # Mask for tokens in this GPU's vocab range
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Adjust indices to local range
            x_local = mask * (x - self.vocab_start_idx)
        else:
            x_local = x
            mask = None

        # Lookup embeddings
        y = F.embedding(x_local, self.weight)

        if self.tp_size > 1:
            # Zero out non-owned tokens and sum across GPUs
            y = mask.unsqueeze(-1) * y
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    Language model head (logits) with vocabulary parallelism.

    Inherits embedding infrastructure but:
    1. Only processes last token in prefill phase
    2. Gathers logits from all GPUs to rank 0
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias, "Bias not supported in LM head"
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for next token prediction.

        In prefill mode, only uses last token of each sequence.
        Gathers partial logits from all GPUs to rank 0.
        """
        context = get_context()

        if context.is_prefill:
            # Extract last token of each sequence
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # Compute partial logits for this GPU's vocab range
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # Gather all partial logits to rank 0
            if self.tp_rank == 0:
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None

            dist.gather(logits, all_logits, dst=0)

            if self.tp_rank == 0:
                # Concatenate partial logits to get full vocab logits
                logits = torch.cat(all_logits, dim=-1)
            else:
                logits = None

        return logits

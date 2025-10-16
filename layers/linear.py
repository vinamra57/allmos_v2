"""
Linear layers with tensor parallelism support.

Design Philosophy:
- Support sharding across multiple GPUs
- Custom weight loading for distributed weights
- Minimal communication overhead

Tensor Parallelism Strategies:
- Column-parallel: Split output dimension (e.g., Q, K, V projections)
- Row-parallel: Split input dimension (e.g., output projection)
- Replicated: No sharding (e.g., small layers)
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion."""
    assert numerator % denominator == 0, \
        f"{numerator} not divisible by {denominator}"
    return numerator // denominator


class LinearBase(nn.Module):
    """Base class for all linear layers with custom weight loading."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """
        Initialize base linear layer.

        Args:
            input_size: Input dimension
            output_size: Output dimension
            bias: Whether to include bias
            tp_dim: Tensor parallel dimension (0 for output, 1 for input, None for replicated)
        """
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom weight loading (to be overridden by subclasses)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (to be overridden by subclasses)."""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Linear layer with replicated weights (no tensor parallelism)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load full weight (no sharding)."""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Linear layer with column-wise parallelism (output dimension sharded)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        Initialize column-parallel linear layer.

        Splits output dimension across GPUs.
        No all-reduce needed (next layer must be row-parallel).
        """
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load sharded weight along output dimension."""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    """Linear layer with row-wise parallelism (input dimension sharded)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        Initialize row-parallel linear layer.

        Splits input dimension across GPUs.
        Requires all-reduce to sum partial results.
        """
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load sharded weight along input dimension."""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    Multiple column-parallel layers merged into one.

    Used for gate_proj + up_proj in FFN.
    Reduces number of kernel launches.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int
    ):
        """Load weight for one of the merged layers."""
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    Merged Q, K, V projections for attention.

    Handles Grouped Query Attention (GQA) where num_kv_heads < num_heads.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str
    ):
        """
        Load weight for Q, K, or V projection.

        Args:
            param: Parameter to load into
            loaded_weight: Weight to load
            loaded_shard_id: One of 'q', 'k', 'v'
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # v
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

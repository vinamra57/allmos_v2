"""
Quantization utilities for FP8 and INT4/INT8.

Implements:
- FP8 (E4M3) weight and activation quantization
- INT4/INT8 KV cache quantization
- Per-tensor and per-channel scaling
- Dynamic activation scaling

References:
- https://arxiv.org/abs/2209.05433 (FP8 for Deep Learning)
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


# FP8 formats supported by NVIDIA GPUs
# E4M3: 4-bit exponent, 3-bit mantissa (better for weights, range: ~-448 to 448)
# E5M2: 5-bit exponent, 2-bit mantissa (better for gradients, range: ~-57344 to 57344)
FP8_E4M3_MAX = 448.0  # Maximum representable value in FP8 E4M3


def quantize_fp8_e4m3(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    per_channel: bool = False,
    channel_dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 E4M3 format.

    Args:
        tensor: Input tensor (BF16/FP16/FP32)
        scale: Pre-computed scale (optional)
        per_channel: Use per-channel quantization (better for weights)
        channel_dim: Dimension for per-channel quantization (default: 0)

    Returns:
        (quantized_tensor, scale)
    """
    if scale is None:
        # Compute scale
        if per_channel:
            # Per-channel: scale per output channel
            # Reduce over all dims except channel_dim
            reduce_dims = [i for i in range(tensor.ndim) if i != channel_dim]
            amax = tensor.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            # Per-tensor: single scale for entire tensor
            amax = tensor.abs().max()

        # Scale = amax / FP8_MAX, ensuring we don't overflow FP8 range
        scale = amax / FP8_E4M3_MAX
        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-10)

    # Quantize: x_fp8 = round(x / scale)
    # PyTorch 2.1+ has native FP8 support via torch.float8_e4m3fn
    scaled = tensor / scale

    if hasattr(torch, 'float8_e4m3fn'):
        # Native FP8 support (PyTorch 2.1+)
        quantized = scaled.to(torch.float8_e4m3fn)
    else:
        # Fallback: clamp to FP8 range and stay in FP16
        # This won't give speedup but maintains correctness
        quantized = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(tensor.dtype)

    return quantized, scale


def dequantize_fp8(
    quantized: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to higher precision.

    Args:
        quantized: FP8 quantized tensor
        scale: Quantization scale

    Returns:
        Dequantized tensor in original dtype
    """
    # x = x_fp8 * scale
    if quantized.dtype == torch.float8_e4m3fn:
        # Convert FP8 to FP16 first, then scale
        return quantized.to(torch.float16) * scale
    else:
        # Already in higher precision
        return quantized * scale


class FP8Linear(nn.Module):
    """
    Linear layer with FP8 weight quantization.

    Stores weights in FP8 E4M3 format for memory savings and speedup on
    supported hardware (Ada Lovelace/Hopper GPUs).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        per_channel_weight: bool = True,
        dynamic_activation: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_channel_weight = per_channel_weight
        self.dynamic_activation = dynamic_activation

        # Initialize weight in FP16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

        # Quantization scales (learned/computed during quantization)
        self.register_buffer('weight_scale', torch.ones(1, dtype=torch.float16))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def quantize_weights(self):
        """
        Quantize weights to FP8 E4M3 (call this after loading pretrained weights).
        """
        with torch.no_grad():
            quantized, scale = quantize_fp8_e4m3(
                self.weight.data,
                per_channel=self.per_channel_weight,
                channel_dim=0  # Output channel
            )
            self.weight.data = quantized
            self.weight_scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP8 computation.

        Args:
            x: Input tensor [batch, in_features]

        Returns:
            Output tensor [batch, out_features]
        """
        # Dequantize weight for computation (temporarily)
        # TODO: Use actual FP8 GEMM when available in PyTorch
        weight_fp16 = dequantize_fp8(self.weight, self.weight_scale)

        if self.dynamic_activation and hasattr(torch, 'float8_e4m3fn'):
            # Quantize activations dynamically (per-token scaling)
            x_quant, x_scale = quantize_fp8_e4m3(x, per_channel=False)
            x_fp16 = dequantize_fp8(x_quant, x_scale)
            output = torch.nn.functional.linear(x_fp16, weight_fp16, self.bias)
        else:
            # Standard FP16 computation
            output = torch.nn.functional.linear(x, weight_fp16, self.bias)

        return output

    @staticmethod
    def from_linear(linear: nn.Linear, **kwargs) -> 'FP8Linear':
        """
        Convert a standard nn.Linear to FP8Linear.

        Args:
            linear: Original linear layer
            **kwargs: Additional FP8Linear arguments

        Returns:
            FP8Linear with copied weights
        """
        fp8_linear = FP8Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            **kwargs
        )

        # Copy weights
        fp8_linear.weight.data.copy_(linear.weight.data.to(torch.float16))
        if linear.bias is not None:
            fp8_linear.bias.data.copy_(linear.bias.data.to(torch.float16))

        # Quantize weights
        fp8_linear.quantize_weights()

        return fp8_linear


# INT4/INT8 Quantization for KV Cache

def quantize_int4(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
    per_channel: bool = True,
    channel_dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT4 (stored in INT8 with 4-bit range).

    Args:
        tensor: Input tensor
        scale: Pre-computed scale (optional)
        zero_point: Pre-computed zero point (optional)
        per_channel: Use per-channel quantization
        channel_dim: Dimension for per-channel quantization

    Returns:
        (quantized_tensor, scale, zero_point)
    """
    if scale is None:
        # Compute scale and zero point
        if per_channel:
            reduce_dims = [i for i in range(tensor.ndim) if i != channel_dim]
            t_min = tensor.amin(dim=reduce_dims, keepdim=True)
            t_max = tensor.amax(dim=reduce_dims, keepdim=True)
        else:
            t_min = tensor.min()
            t_max = tensor.max()

        # INT4 range: -8 to 7 (symmetric) or 0 to 15 (asymmetric)
        # Using symmetric for simplicity: -8 to 7
        scale = (t_max - t_min) / 15.0  # 15 = 2^4 - 1
        scale = torch.clamp(scale, min=1e-10)
        zero_point = torch.round(-t_min / scale)
        zero_point = torch.clamp(zero_point, 0, 15)

    # Quantize
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, 0, 15).to(torch.int8)

    return quantized, scale, zero_point


def dequantize_int4(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize INT4 tensor.

    Args:
        quantized: INT4 quantized tensor (stored as INT8)
        scale: Quantization scale
        zero_point: Quantization zero point
        dtype: Target dtype

    Returns:
        Dequantized tensor
    """
    return (quantized.to(dtype) - zero_point) * scale


def quantize_int8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    per_channel: bool = True,
    channel_dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 (symmetric).

    Args:
        tensor: Input tensor
        scale: Pre-computed scale (optional)
        per_channel: Use per-channel quantization
        channel_dim: Dimension for per-channel quantization

    Returns:
        (quantized_tensor, scale)
    """
    if scale is None:
        if per_channel:
            reduce_dims = [i for i in range(tensor.ndim) if i != channel_dim]
            amax = tensor.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            amax = tensor.abs().max()

        scale = amax / 127.0  # INT8 symmetric range: -127 to 127
        scale = torch.clamp(scale, min=1e-10)

    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)

    return quantized, scale


def dequantize_int8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize INT8 tensor.

    Args:
        quantized: INT8 quantized tensor
        scale: Quantization scale
        dtype: Target dtype

    Returns:
        Dequantized tensor
    """
    return quantized.to(dtype) * scale


def replace_linear_with_fp8(
    model: nn.Module,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Replace all linear layers in model with FP8Linear.

    Args:
        model: Model to modify
        target_modules: List of module names to replace (None = all)

    Returns:
        Modified model
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # Recursive replacement
            replace_linear_with_fp8(module, target_modules)

        if isinstance(module, nn.Linear):
            if target_modules is None or name in target_modules:
                # Replace with FP8Linear
                fp8_layer = FP8Linear.from_linear(module)
                setattr(model, name, fp8_layer)

    return model

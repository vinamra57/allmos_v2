"""
Model weight loader for HuggingFace safetensors checkpoints.

Design Philosophy:
- Support packed weight loading (QKV, gate_up_proj merged layers)
- Custom weight loaders for tensor parallelism sharding
- Efficient loading from safetensors format

Weight Loading Strategy:
1. Iterate through all safetensors files in model directory
2. For each weight, check if it needs special handling (packed modules)
3. Call custom weight_loader if available (for sharding)
4. Otherwise use default copy operation

Packed Modules:
Models often merge related weights for efficiency:
- q_proj, k_proj, v_proj → qkv_proj (merged QKV)
- gate_proj, up_proj → gate_up_proj (merged FFN)

The packed_modules_mapping in the model tells us how to map
checkpoint weights to our merged parameters.
"""
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    Default weight loading: simple copy.

    Args:
        param: Parameter to load into
        loaded_weight: Weight from checkpoint
    """
    assert param.data.shape == loaded_weight.shape, \
        f"Shape mismatch: param {param.data.shape} vs loaded {loaded_weight.shape}"
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, model_path: str) -> None:
    """
    Load model weights from HuggingFace safetensors checkpoint.

    This function:
    1. Finds all *.safetensors files in model_path
    2. For each weight in the checkpoint:
       - Checks if it's part of a packed module (e.g., q_proj → qkv_proj)
       - Looks up the corresponding parameter in the model
       - Calls the parameter's custom weight_loader (if exists)
       - Otherwise uses default copy operation

    Args:
        model: PyTorch model to load weights into
        model_path: Path to model directory containing *.safetensors files

    Example:
        >>> model = Qwen3ForCausalLM(config)
        >>> load_model(model, "~/huggingface/Qwen3-0.6B/")

    Packed Module Handling:
        If model has packed_modules_mapping = {"q_proj": ("qkv_proj", "q")},
        then checkpoint weight "layer.0.self_attn.q_proj.weight" will be loaded
        into parameter "layer.0.self_attn.qkv_proj.weight" with shard_id="q".
    """
    # Get packed module mapping (if model defines it)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # Find all safetensors files
    safetensors_files = glob(os.path.join(model_path, "*.safetensors"))
    assert safetensors_files, f"No safetensors files found in {model_path}"

    print(f"Loading weights from {len(safetensors_files)} safetensors file(s)...")

    # Load weights from each file
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                # Check if this weight is part of a packed module
                param_name = weight_name
                shard_id = None

                for checkpoint_key, (merged_key, shard) in packed_modules_mapping.items():
                    if checkpoint_key in weight_name:
                        # This weight should be loaded into a merged parameter
                        param_name = weight_name.replace(checkpoint_key, merged_key)
                        shard_id = shard
                        break

                # Get the parameter from the model
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    print(f"Warning: Parameter {param_name} not found in model, skipping")
                    continue

                # Load the weight tensor
                loaded_weight = f.get_tensor(weight_name)

                # Get weight loader (custom or default)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                # Call weight loader with appropriate arguments
                try:
                    if shard_id is not None:
                        # Packed module: pass shard_id
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        # Regular parameter: just pass weight
                        weight_loader(param, loaded_weight)
                except Exception as e:
                    print(f"Error loading {weight_name} into {param_name}: {e}")
                    raise

    print("Weight loading complete!")

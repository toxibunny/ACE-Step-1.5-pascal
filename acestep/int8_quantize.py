"""Manual int8 weight-only quantization for Pascal GPUs.

Stores Linear layer weights as int8 (4x smaller than fp32),
dequantizes to fp32 during forward pass. No extra dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with int8 weight storage."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features

        # Quantize weight: [out, in] -> int8 [out, in]
        w = original.weight.data.float()
        self.scale = w.abs().max() / 127.0
        if self.scale.item() == 0:
            self.scale = torch.tensor(1.0)
        self.weight_int8 = (w / self.scale).clamp(-127, 127).round().to(torch.int8)

        # Bias stays fp32
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.float(), requires_grad=False)
        else:
            self.bias = None

        # Freeze
        self.weight_int8 = self.weight_int8.detach()
        self.scale = self.scale.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: int8 * scale -> fp32
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"quantized=int8")


def quantize_module_int8(module: nn.Module, filter_fn=None) -> int:
    """Replace all nn.Linear layers in a module with Int8Linear.
    
    Args:
        module: The model (or sub-module) to quantize.
        filter_fn: Optional callable(module, fqn) -> bool. If provided,
                   only quantize layers where it returns True.
    
    Returns:
        Number of layers quantized.
    """
    count = 0
    replacements = []

    for name, child in module.named_children():
        # Recurse first
        count += quantize_module_int8(child, filter_fn)

        # Check if this child is a Linear we should quantize
        if isinstance(child, nn.Linear):
            fqn = name  # relative name for filter
            if filter_fn is None or filter_fn(child, fqn):
                parent = module
                replacements.append((parent, name, child))

    # Do replacements after traversal to avoid mutation during iteration
    for parent, name, linear in replacements:
        int8_layer = Int8Linear(linear).to(linear.weight.device)
        setattr(parent, name, int8_layer)
        count += 1

    return count


def quantize_dit_int8(model: nn.Module) -> int:
    """Quantize the DiT decoder linear layers to int8.
    
    Only quantizes layers under 'decoder' that are not tokenizers.
    """
    def filter_fn(module, fqn):
        # fqn is relative to the parent being traversed
        # We want to quantize all Linears in the decoder
        return True

    if not hasattr(model, 'decoder'):
        # Try to find the decoder
        quantized = 0
        for name, child in model.named_children():
            if 'decoder' in name.lower() or 'dit' in name.lower() or 'transformer' in name.lower():
                quantized += quantize_module_int8(child, filter_fn)
        return quantized

    return quantize_module_int8(model.decoder, filter_fn)

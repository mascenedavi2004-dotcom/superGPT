"""
FP8 Mixed-Precision Training Utilities
========================================
Enables FP8 (8-bit floating point) training for 2× memory reduction
and faster matmul on Hopper+ GPUs (H100, H800, B200).

Uses PyTorch's native FP8 support (torch.float8_e4m3fn for forward,
torch.float8_e5m2 for backward) with per-tensor dynamic scaling.

Reference: DeepSeek-V3 Technical Report, Section 3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# FP8 dtypes (available in PyTorch 2.1+)
try:
    FP8_E4M3 = torch.float8_e4m3fn  # 4-bit mantissa: better precision
    FP8_E5M2 = torch.float8_e5m2    # 5-bit exponent: better range (grads)
    FP8_AVAILABLE = hasattr(torch, '_scaled_mm')
except AttributeError:
    FP8_E4M3 = None
    FP8_E5M2 = None
    FP8_AVAILABLE = False


class FP8ScaleTracker:
    """Dynamic per-tensor scale tracking for FP8 quantization.

    Maintains a running maximum of tensor values to compute
    optimal scaling factors. Uses delayed scaling (scale from
    previous iteration) for stability.
    """

    def __init__(self, history_len: int = 16):
        self.history_len = history_len
        self.amax_history = []
        self.scale = 1.0

    def update(self, tensor: torch.Tensor) -> float:
        """Update scale based on tensor statistics."""
        with torch.no_grad():
            amax = tensor.abs().max().item()
            self.amax_history.append(amax)
            if len(self.amax_history) > self.history_len:
                self.amax_history.pop(0)

            # Use max of history for delayed scaling
            max_amax = max(self.amax_history) if self.amax_history else 1.0
            max_amax = max(max_amax, 1e-12)  # Prevent /0

            # E4M3 max representable value: 448.0
            fp8_max = 448.0
            self.scale = fp8_max / max_amax
            return self.scale


def quantize_to_fp8(
    tensor: torch.Tensor,
    scale: float = 1.0,
    fp8_dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with per-tensor scaling.

    Args:
        tensor: Input tensor (bf16/fp32)
        scale: Scaling factor (tensor * scale → FP8 range)
        fp8_dtype: Target FP8 dtype (default: e4m3fn)

    Returns:
        fp8_tensor: Quantized tensor
        scale_inv: Inverse scale for dequantization
    """
    if fp8_dtype is None:
        fp8_dtype = FP8_E4M3

    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=tensor.device)
    scaled = tensor.float() * scale
    fp8_tensor = scaled.to(fp8_dtype)
    scale_inv = torch.tensor(1.0 / scale, dtype=torch.float32, device=tensor.device)
    return fp8_tensor, scale_inv


def fp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 matrix multiplication using torch._scaled_mm.

    Computes a @ b.T (like F.linear: input @ weight.T)

    Args:
        a: FP8 tensor (M, K)
        b: FP8 tensor (N, K) — weight layout, will be transposed
        a_scale, b_scale: Per-tensor inverse scales
        out_dtype: Output dtype

    Returns:
        Result tensor in out_dtype — shape (M, N)
    """
    if not FP8_AVAILABLE:
        # Fallback: dequantize and use standard matmul
        a_f = a.to(out_dtype) * a_scale
        b_f = b.to(out_dtype) * b_scale
        return torch.matmul(a_f, b_f.t())

    # torch._scaled_mm: (M,K) @ (K,N) — need to transpose b from (N,K) to (K,N)
    return torch._scaled_mm(
        a, b.t(),
        scale_a=a_scale,
        scale_b=b_scale,
        out_dtype=out_dtype,
    )


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with FP8 computation.

    Keeps master weights in bf16/fp32, quantizes to FP8 for the
    forward pass matmul. Gradients flow through in higher precision.

    This gives ~2× memory reduction for activations and ~1.5× speedup
    for matmuls on Hopper GPUs.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Master weights in bf16
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        nn.init.kaiming_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        # Scale trackers
        self.input_scale_tracker = FP8ScaleTracker()
        self.weight_scale_tracker = FP8ScaleTracker()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not FP8_AVAILABLE or not self.training:
            # Standard bf16 path (inference or no FP8 support)
            return F.linear(x.to(self.weight.dtype), self.weight, self.bias)

        # FP8 forward pass
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # Quantize input and weight to FP8
        x_scale = self.input_scale_tracker.update(x_2d)
        w_scale = self.weight_scale_tracker.update(self.weight)

        x_fp8, x_scale_inv = quantize_to_fp8(x_2d, x_scale)
        w_fp8, w_scale_inv = quantize_to_fp8(self.weight, w_scale)

        # FP8 matmul: x_fp8 @ w_fp8.T (w_fp8 is in (out, in) layout)
        out = fp8_matmul(x_fp8, w_fp8, x_scale_inv, w_scale_inv)
        out = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out


def convert_model_to_fp8(model: nn.Module, skip_patterns: list = None) -> nn.Module:
    """Convert nn.Linear layers to FP8Linear for FP8 training.

    Args:
        model: The model to convert
        skip_patterns: List of name patterns to skip (e.g., ['lm_head', 'wte'])

    Returns:
        Modified model with FP8 linear layers
    """
    if not FP8_AVAILABLE:
        print("  Warning: FP8 not available (requires PyTorch 2.1+ with Hopper GPU)")
        print("  Falling back to standard precision")
        return model

    skip_patterns = skip_patterns or ['wte', 'wpe', 'ln_', 'norm']
    converted = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip embeddings and norms
            if any(p in name for p in skip_patterns):
                continue

            # Replace with FP8 variant
            fp8_linear = FP8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            fp8_linear.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                fp8_linear.bias.data = module.bias.data.to(torch.bfloat16)

            # Set the attribute on the parent module
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name

            setattr(parent, attr_name, fp8_linear)
            converted += 1

    print(f"  Converted {converted} layers to FP8")
    return model

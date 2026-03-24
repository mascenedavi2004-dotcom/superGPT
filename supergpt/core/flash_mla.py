"""
FlashMLA Integration — Optional CUDA Kernel Acceleration
==========================================================
Provides a transparent interface to DeepSeek's FlashMLA CUDA kernels
for MLA attention. Falls back to PyTorch implementation if not available.

FlashMLA achieves 660 TFLOPS on H800 SXM5 (vs ~50 TFLOPS with PyTorch).
Requires: SM90+ GPU (H100/H800/B200), CUDA 12.8+, flash_mla package.

Install FlashMLA:
  git clone https://github.com/deepseek-ai/FlashMLA.git
  cd FlashMLA && pip install -v .

Usage in superGPT:
  Automatic — if flash_mla is installed, MLA attention auto-routes to CUDA.
"""

import torch
from typing import Optional, Tuple

# Try to import FlashMLA CUDA kernels
FLASH_MLA_AVAILABLE = False
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    FLASH_MLA_AVAILABLE = True
except ImportError:
    pass


def get_mla_backend() -> str:
    """Return the active MLA backend name."""
    if FLASH_MLA_AVAILABLE:
        return "flash_mla (CUDA)"
    return "pytorch (naive/absorbed)"


def flash_mla_decode(
    q: torch.Tensor,              # (B, n_heads, 1, qk_head_dim)
    kv_cache: torch.Tensor,       # (B, max_seq_len, kv_lora_rank + qk_rope_dim)
    cache_seqlens: torch.Tensor,  # (B,) — actual sequence lengths
    block_table: torch.Tensor,    # Block table for paged attention
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:
    """Run FlashMLA decode kernel if available.

    This handles the case where we're generating one token at a time
    with a KV-cache. FlashMLA is optimized for this decode path.

    Returns:
        output: (B, n_heads, 1, v_head_dim) attention output

    Falls back to None if FlashMLA is not available (caller should
    use the PyTorch absorbed/naive path instead).
    """
    if not FLASH_MLA_AVAILABLE:
        return None

    try:
        # FlashMLA expects specific tensor layouts
        # Reshape q for FlashMLA: (B, n_heads, 1, head_dim) → (B, n_heads, head_dim)
        B, n_heads, _, head_dim = q.shape
        q_squeezed = q.squeeze(2)  # (B, n_heads, head_dim)

        # Get metadata for block-sparse attention
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens,
            block_table.shape[-1],  # num_blocks_per_seq
        )

        # Run FlashMLA kernel
        output, _ = flash_mla_with_kvcache(
            q_squeezed,
            kv_cache,
            block_table,
            cache_seqlens,
            kv_lora_rank,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale=softmax_scale,
        )

        return output.unsqueeze(2)  # (B, n_heads, 1, v_head_dim)

    except Exception as e:
        # Any error: fall back to PyTorch path
        print(f"  FlashMLA error (falling back to PyTorch): {e}")
        return None


def flash_mla_prefill(
    q: torch.Tensor,              # (B, n_heads, T, qk_head_dim)
    kv_latent: torch.Tensor,      # (B, T, kv_lora_rank)
    k_rope: torch.Tensor,         # (B, 1, T, qk_rope_head_dim)
    softmax_scale: float,
    causal: bool = True,
) -> Optional[torch.Tensor]:
    """Run FlashMLA prefill kernel if available.

    For the prefill (training) path with full sequence.
    Falls back to None if not available.
    """
    if not FLASH_MLA_AVAILABLE:
        return None

    # FlashMLA prefill support depends on version
    # For now, return None to use PyTorch naive path for prefill
    # (FlashMLA prefill is optimized for SM100+ / B200)
    return None


def print_flash_mla_info():
    """Print FlashMLA availability and info."""
    if FLASH_MLA_AVAILABLE:
        print("  FlashMLA: ✅ Available (CUDA kernels)")
        print("  Expected: 660 TFLOPS (H800), 1450 TFLOPS (B200)")
    else:
        print("  FlashMLA: ❌ Not installed (using PyTorch)")
        print("  Install: git clone https://github.com/deepseek-ai/FlashMLA.git")
        print("           cd FlashMLA && pip install -v .")

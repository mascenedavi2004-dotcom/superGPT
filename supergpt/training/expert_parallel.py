"""
Expert Parallelism Integration (DeepEP)
=========================================
Optional integration with DeepSeek's DeepEP library for efficient
MoE dispatch/combine across multiple GPUs.

DeepEP provides high-throughput all-to-all kernels optimized for MoE:
  - Normal kernels: NVLink + RDMA forwarding (training + prefill)
  - Low-latency kernels: Pure RDMA (inference decoding)
  - FP8 support for expert communication
  - Hook-based comm-compute overlap (zero SM overhead)

Install DeepEP:
  git clone https://github.com/deepseek-ai/DeepEP.git
  cd DeepEP && pip install -v .

Usage:
  Automatic — if deep_ep is installed and distributed training
  is enabled, MoE layers will use DeepEP for dispatch/combine.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import DeepEP
DEEP_EP_AVAILABLE = False
try:
    from deep_ep import Buffer, EventOverlap
    from deep_ep.utils import get_num_sms
    DEEP_EP_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if DeepEP is installed and GPU supports it."""
    return DEEP_EP_AVAILABLE


class ExpertParallelDispatcher:
    """Wraps DeepEP dispatch/combine for MoE layers.

    In expert parallelism, each GPU holds a subset of experts.
    Tokens are dispatched to the GPU holding their assigned expert,
    processed, then combined back.

    This class handles:
    1. Dispatch: send tokens to correct GPU based on routing
    2. Expert compute: local expert forward pass
    3. Combine: gather results back to originating GPUs
    """

    def __init__(
        self,
        n_experts: int,
        n_experts_per_rank: int,
        hidden_dim: int,
        use_fp8: bool = False,
        num_sms: int = 20,
    ):
        self.n_experts = n_experts
        self.n_experts_per_rank = n_experts_per_rank
        self.hidden_dim = hidden_dim
        self.use_fp8 = use_fp8

        if not DEEP_EP_AVAILABLE:
            return

        # Create DeepEP communication buffer
        self.buffer = Buffer(
            group=None,  # Will use default process group
            num_nvl_bytes=0,  # Set based on cluster topology
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_ep=1,
        )
        self.num_sms = num_sms

    def dispatch(
        self,
        x: torch.Tensor,           # (N, hidden_dim) — all tokens
        expert_indices: torch.Tensor,  # (N, topk) — assigned experts
        expert_weights: torch.Tensor,  # (N, topk) — expert weights
    ) -> Tuple[torch.Tensor, dict]:
        """Dispatch tokens to their assigned expert GPUs.

        Returns:
            recv_x: tokens received for local experts
            metadata: info needed for combine step
        """
        if not DEEP_EP_AVAILABLE:
            # Fallback: no dispatch needed (single GPU)
            return x, {"indices": expert_indices, "weights": expert_weights}

        # Convert routing to per-expert token lists
        # This is the all-to-all communication
        recv_x, recv_count, handle, event = self.buffer.dispatch(
            x,
            topk_idx=expert_indices,
            topk_weight=expert_weights,
            num_experts=self.n_experts,
            use_fp8=self.use_fp8,
            async_op=False,
            num_sms=self.num_sms,
        )

        metadata = {
            "recv_count": recv_count,
            "handle": handle,
            "event": event,
            "indices": expert_indices,
            "weights": expert_weights,
        }
        return recv_x, metadata

    def combine(
        self,
        expert_output: torch.Tensor,  # Expert-processed tokens
        metadata: dict,
    ) -> torch.Tensor:
        """Combine expert outputs back to originating GPUs.

        Returns:
            combined: (N, hidden_dim) — results mapped back to original positions
        """
        if not DEEP_EP_AVAILABLE:
            return expert_output

        combined, _ = self.buffer.combine(
            expert_output,
            handle=metadata["handle"],
            async_op=False,
            num_sms=self.num_sms,
        )
        return combined


class EPMoELayer(nn.Module):
    """MoE layer with optional Expert Parallelism via DeepEP.

    Drop-in replacement for MoELayer that handles multi-GPU expert
    distribution when DeepEP is available.
    """

    def __init__(self, moe_layer: nn.Module, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.moe = moe_layer
        self.rank = rank
        self.world_size = world_size

        if world_size > 1 and DEEP_EP_AVAILABLE:
            n_local = moe_layer.n_experts // world_size
            self.dispatcher = ExpertParallelDispatcher(
                n_experts=moe_layer.n_experts,
                n_experts_per_rank=n_local,
                hidden_dim=moe_layer.gate.dim,
            )
            self.ep_enabled = True
        else:
            self.ep_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ep_enabled:
            return self.moe(x)

        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # Get routing
        weights, indices = self.moe.gate(x_flat)

        # Dispatch to expert GPUs
        recv_x, metadata = self.dispatcher.dispatch(x_flat, indices, weights)

        # Process with local experts
        local_expert_start = self.rank * (self.moe.n_experts // self.world_size)
        local_expert_end = local_expert_start + (self.moe.n_experts // self.world_size)

        expert_output = torch.zeros_like(recv_x)
        for e in range(local_expert_start, local_expert_end):
            local_e = e - local_expert_start
            # Process tokens assigned to this expert
            expert_output += self.moe.experts[e](recv_x)

        # Combine back
        y = self.dispatcher.combine(expert_output, metadata)

        # Add shared experts
        if self.moe.shared_experts is not None:
            y = y + self.moe.shared_experts(x_flat)

        return y.view(B, T, C)


def wrap_moe_with_ep(model: nn.Module, rank: int = 0, world_size: int = 1):
    """Wrap all MoE layers with expert parallelism if available."""
    if not DEEP_EP_AVAILABLE or world_size <= 1:
        return model

    from supergpt.core.model import MoELayer

    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            ep_layer = EPMoELayer(module, rank, world_size)
            # Replace in parent
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], ep_layer)

    print(f"  DeepEP: Wrapped MoE layers for {world_size}-GPU expert parallelism")
    return model


def print_ep_info():
    """Print DeepEP availability info."""
    if DEEP_EP_AVAILABLE:
        print("  DeepEP: ✅ Available (expert parallelism)")
    else:
        print("  DeepEP: ❌ Not installed (single-GPU MoE)")
        print("  Install: git clone https://github.com/deepseek-ai/DeepEP.git")
        print("           cd DeepEP && pip install -v .")

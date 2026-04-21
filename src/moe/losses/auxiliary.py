"""MoE auxiliary losses: extra terms on the task loss to stabilize routing and experts.

This module is the single entry point for combining:
- :func:`load_balance_loss` — uniform expert use (Switch Transformer).
- :func:`router_z_loss` — dampens extreme router logits (ST-MoE).

The main cross-entropy (or other task) loss lives in the training loop; add
``total_aux`` from :func:`moe_auxiliary_loss` to that loss with the configured weights.
"""

from __future__ import annotations

import torch

from .load_balance import load_balance_loss
from .router_z_loss import router_z_loss


def moe_auxiliary_loss(
    router_logits: torch.Tensor,
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    n_experts: int,
    *,
    load_balance_weight: float = 0.01,
    router_z_weight: float = 0.001,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of MoE auxiliary losses.

    router_logits: (BT, n_experts) pre-softmax router outputs (for z-loss).
    router_probs:  (BT, n_experts) softmax(router_logits) (for load-balance loss).
    topk_idx:      (BT, k) indices of experts selected per token.
    """
    lb = load_balance_loss(router_probs, topk_idx, n_experts)
    z = router_z_loss(router_logits)
    total = load_balance_weight * lb + router_z_weight * z
    parts = {
        "load_balance": lb.detach(),
        "router_z": z.detach(),
        "weighted_load_balance": (load_balance_weight * lb).detach(),
        "weighted_router_z": (router_z_weight * z).detach(),
    }
    return total, parts

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterOutput(NamedTuple):
    """Stable field order so callers don't rely on positional unpacking."""

    topk_idx: torch.Tensor  # (BT, k)
    topk_weights: torch.Tensor  # (BT, k), rows sum to 1
    full_probs: torch.Tensor  # (BT, n_experts)
    logits: torch.Tensor  # (BT, n_experts)


class TopKRouter(nn.Module):
    """Top-k softmax router. Returns the chosen expert indices, the
    renormalized routing weights, and the full routing probabilities
    (which the load-balance loss needs)."""

    def __init__(self, d_model: int, n_experts: int, k: int):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x_flat: torch.Tensor) -> RouterOutput:
        # x_flat: (BT, d_model) — a flat batch of tokens
        logits = self.gate(x_flat)  # (BT, n_experts)

        # Full softmax over ALL experts — needed for the load-balance loss
        full_probs = F.softmax(logits, dim=-1)  # (BT, n_experts)

        # Top-k selection then renormalize over the selected k
        topk_logits, topk_idx = logits.topk(self.k, dim=-1)  # both (BT, k)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (BT, k), rows sum to 1

        return RouterOutput(topk_idx, topk_weights, full_probs, logits)

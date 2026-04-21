import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """A single expert: a SwiGLU feed-forward network. Identical in shape
    to a LLaMA / Mixtral FFN — what makes it an 'expert' is that there are
    N of them and a router decides which ones run for each token."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate-up branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # value-up branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) — works on (B, T, d) or flattened (BT, d)
        gate = F.silu(self.w1(x))  # (..., d_ff)
        value = self.w3(x)  # (..., d_ff)
        return self.w2(gate * value)  # (..., d_model)

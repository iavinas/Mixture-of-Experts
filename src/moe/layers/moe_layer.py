import torch
import torch.nn as nn

from .expert_ffn import SwiGLUExpert
from .router import TopKRouter


class MoELayer(nn.Module):
    """Routes each token to its top-k experts and combines outputs.

    Dispatch: flatten (token, slot) assignments, sort by expert id so each
    expert runs on a contiguous slice (one gather + one scatter). Avoids the
    per-expert boolean scan of the loop-mask variant.
    """

    def __init__(
        self, d_model: int, d_ff: int, n_experts: int, k: int, n_shared: int = 0
    ):
        super().__init__()
        assert k <= n_experts, "k cannot exceed n_experts"
        self.n_experts = n_experts
        self.k = k
        self.router = TopKRouter(d_model, n_experts, k)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUExpert(d_model, d_ff) for _ in range(n_shared)]
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        BT = B * T
        x_flat = x.reshape(BT, D)

        router_out = self.router(x_flat)
        topk_idx = router_out.topk_idx  # (BT, k)
        topk_w = router_out.topk_weights  # (BT, k)

        # Flatten assignments into (BT*k,) then sort by expert id so we can
        # run each expert on a contiguous slice of tokens.
        flat_exp = topk_idx.reshape(-1)  # (BT*k,)
        # token_id[i] = which source token slot i belongs to
        token_id = (
            torch.arange(BT, device=x.device)
            .unsqueeze(1)
            .expand(BT, self.k)
            .reshape(-1)
        )
        flat_w = topk_w.reshape(-1)

        sort_vals, sort_idx = flat_exp.sort()
        tok_sorted = token_id[sort_idx]
        w_sorted = flat_w[sort_idx]

        # Split points per expert — bincount gives counts, cumsum gives offsets.
        counts = torch.bincount(sort_vals, minlength=self.n_experts).tolist()

        out = torch.zeros_like(x_flat)
        offset = 0
        for expert_id, count in enumerate(counts):
            if count == 0:
                offset += count
                continue
            sl = slice(offset, offset + count)
            tok_idx = tok_sorted[sl]
            tokens = x_flat.index_select(0, tok_idx)
            weights = w_sorted[sl].unsqueeze(-1)
            expert_out = self.experts[expert_id](tokens) * weights
            out.index_add_(0, tok_idx, expert_out)
            offset += count

        # Shared experts run on every token, unweighted (DeepSeek-style).
        for sh in self.shared:
            out = out + sh(x_flat)

        return (
            out.reshape(B, T, D),
            router_out.full_probs,
            router_out.logits,
            topk_idx,
        )

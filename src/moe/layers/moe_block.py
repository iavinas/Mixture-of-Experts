import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .expert_ffn import SwiGLUExpert
from .moe_layer import MoELayer
from .norm import RMSNorm
from .rope import RotaryEmbedding


class MoEBlock(nn.Module):
    """Pre-norm transformer block; FFN is either dense SwiGLU or a MoE layer.
    Always returns (x, new_cache, router_stats); router_stats is None for dense FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int,
        k: int,
        n_shared: int,
        use_moe: bool,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=attn_dropout)
        self.norm2 = RMSNorm(d_model)
        self.use_moe = use_moe
        self.ffn = (
            MoELayer(d_model, d_ff, n_experts, k, n_shared)
            if use_moe
            else SwiGLUExpert(d_model, d_ff)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # Attention sublayer.
        h, new_cache = self.attn(self.norm1(x), rope, kv_cache)
        x = x + h

        # FFN sublayer.
        h_in = self.norm2(x)
        if self.use_moe:
            ffn_out, full_probs, logits, topk_idx = self.ffn(h_in)
            router_stats = (full_probs, logits, topk_idx)
        else:
            ffn_out = self.ffn(h_in)
            router_stats = None
        x = x + ffn_out
        return x, new_cache, router_stats

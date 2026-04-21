import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rope

# Finite mask bias: MPS (and some long-sequence paths) can yield NaNs from
# softmax(-inf * ...) mixing; GPT-style models often use ~-1e4 / -1e9 instead.
_ATTN_MASK_BIAS = -10_000.0


def _math_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    past_len: int,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Reference SDPA: stable on backends where :func:`F.scaled_dot_product_attention`
    returns NaNs (notably some MPS + causal cases).
    Shapes: ``q`` (B, H, Tq, D), ``k``/``v`` (B, H, Tk, D).
    """
    orig_dtype = q.dtype
    # MPS matmul/softmax over long T is more stable in float32.
    if q.device.type == "mps":
        q = q.float()
        k = k.float()
        v = v.float()

    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (d**-0.5)
    if is_causal:
        tq, tk = q.size(2), k.size(2)
        qi = torch.arange(tq, device=q.device, dtype=torch.long).view(tq, 1)
        kj = torch.arange(tk, device=q.device, dtype=torch.long).view(1, tk)
        scores = scores.masked_fill(
            kj > past_len + qi,
            torch.tensor(_ATTN_MASK_BIAS, device=q.device, dtype=scores.dtype),
        )
    scores = scores - scores.amax(dim=-1, keepdim=True)
    attn = torch.softmax(scores, dim=-1)
    if training and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    out = torch.matmul(attn, v)
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


def _use_math_attention(device: torch.device) -> bool:
    if device.type == "mps":
        return True
    return os.environ.get("MOE_FORCE_MATH_ATTENTION", "").strip() == "1"


class CausalSelfAttention(nn.Module):
    """Causal MHA with RoPE, KV cache, correct position tracking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Rotate q and k with their *true* positions in the sequence.
        pos = torch.arange(past_len, past_len + T, device=x.device)
        cos, sin = rope(pos)
        q, k = apply_rope(q, k, cos, sin)

        # Extend the cache. Note: cached k was already RoPE-rotated at its own position.
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # Causal mask is only needed when T > 1 (prefill or training).
        # In single-token decode, every cached key is already in the new query's past.
        # NOTE: the fast path (F.scaled_dot_product_attention(..., is_causal=True))
        # applies a lower-triangular mask over the *full* concatenated key length,
        # which is only correct when past_len == 0 (fresh prefill or pure training).
        # Reusing a non-empty cache with T > 1 would require an explicit mask; we
        # reject that combination rather than silently miscompute.
        if T > 1 and past_len > 0:
            raise ValueError(
                "Multi-token forward with a non-empty KV cache is not supported; "
                "either start from an empty cache or feed one token at a time."
            )
        is_causal = T > 1
        dropout_p = self.dropout_p if self.training else 0.0
        if _use_math_attention(q.device):
            out = _math_scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                past_len=past_len,
                dropout_p=dropout_p,
                training=self.training,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, dropout_p=dropout_p
            )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out), new_cache

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables held as buffers. One instance per model, shared across layers."""  # noqa: E501

    def __init__(self, d_head: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, inv_freq)  # (S, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, d_head), Llama-style halving
        # Non-persistent: not saved in state_dict; rebuilt from config on load.
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (T,); returns (T, d_head) each, on the right device.
        return self.cos[position_ids], self.sin[position_ids]


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, H, T, d_head);  cos, sin: (T, d_head)."""

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # Match q/k dtype so mixed precision does not silently upcast.
    cos = cos.to(q.dtype)[None, None, :, :]
    sin = sin.to(q.dtype)[None, None, :, :]
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)

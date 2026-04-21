import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.moe_block import MoEBlock
from ..layers.norm import RMSNorm
from ..layers.rope import RotaryEmbedding


class MoELM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 6,
        n_experts: int = 8,
        k: int = 2,
        n_shared: int = 1,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_experts = n_experts

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # One rope instance shared across all layers; tables live as buffers.
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len, base=rope_base)

        # Alternate: even layers dense, odd layers MoE (DeepSeek-style interleaving).
        self.blocks = nn.ModuleList(
            [
                MoEBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    n_experts,
                    k,
                    n_shared,
                    use_moe=True,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, kv_caches: list | None = None):
        # tokens: (B, T) int64
        B, T = tokens.shape
        past_len = 0 if kv_caches is None else kv_caches[0][0].shape[2]
        if past_len + T > self.max_seq_len:
            raise ValueError(
                f"sequence length {past_len + T} exceeds model.max_seq_len "
                f"({self.max_seq_len}); raise max_seq_len in the model config or "
                f"reduce data.seq_len / generation length."
            )

        x = self.tok_emb(tokens)  # (B, T, d_model)

        new_caches, routing = [], []
        for i, block in enumerate(self.blocks):
            cache_in = kv_caches[i] if kv_caches is not None else None
            x, new_cache, route_info = block(x, self.rope, cache_in)
            new_caches.append(new_cache)
            if route_info is not None:
                routing.append(route_info)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits, new_caches, routing

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Simple greedy/top-k sampling with KV cache."""
        self.eval()
        # Prefill.
        logits, kv_caches, _ = self(prompt, kv_caches=None)
        out = [prompt]

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, k=top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            out.append(next_tok)
            # Decode step: feed just the new token.
            logits, kv_caches, _ = self(next_tok, kv_caches=kv_caches)

        return torch.cat(out, dim=1)

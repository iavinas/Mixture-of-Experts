"""MoE LM hyperparameters.

Load from YAML with keys matching field names (snake_case).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MoELMConfig:
    """Configuration for :class:`~moe.models.moe_lm.MoELM`.

    Example YAML (``configs/model/toy_moe_transformer.yaml``):

    .. code-block:: yaml

        vocab_size: 32000
        d_model: 256
        n_heads: 4
        d_ff: 1024
        n_layers: 6
        n_experts: 8
        k: 2
        n_shared: 1
        max_seq_len: 512
        rope_base: 10000.0

    Unknown keys in the file are ignored so you can nest this document under
    a larger experiment config and pass only the model mapping to :meth:`from_dict`.
    """

    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 6
    n_experts: int = 8
    k: int = 2
    n_shared: int = 1
    max_seq_len: int = 512
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        d_head = self.d_model // self.n_heads
        if d_head % 2 != 0:
            # RoPE's rotate_half splits d_head in two equal halves.
            raise ValueError(
                f"d_head (d_model/n_heads = {d_head}) must be even for RoPE"
            )
        if self.d_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {self.d_ff}")
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {self.n_experts}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.k > self.n_experts:
            raise ValueError(f"k ({self.k}) cannot exceed n_experts ({self.n_experts})")
        if self.n_shared < 0:
            raise ValueError(f"n_shared must be >= 0, got {self.n_shared}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoELMConfig:
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, *, section: str | None = None) -> MoELMConfig:
        """Load from a YAML file. If ``section`` is set, read ``data[section]``."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"YAML file is empty: {path}")
        if section is not None:
            if not isinstance(raw, dict) or section not in raw:
                raise KeyError(f"Missing key {section!r} in {path}")
            raw = raw[section]
        if not isinstance(raw, dict):
            raise TypeError(
                f"YAML must map to an object, got {type(raw).__name__} (path={path})"
            )
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(
        self,
        path: str | Path,
        *,
        section: str | None = None,
        default_flow_style: bool = False,
    ) -> None:
        """Write YAML. If ``section`` is set, wrap the mapping as ``{section: ...}``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = (
            {section: self.to_dict()} if section is not None else self.to_dict()
        )
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                payload,
                f,
                default_flow_style=default_flow_style,
                allow_unicode=True,
                sort_keys=False,
            )

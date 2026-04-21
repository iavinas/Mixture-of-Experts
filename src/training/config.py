"""Training / optimizer step settings (YAML-loadable)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class OptimizerConfig:
    """AdamW + warmup/cosine schedule settings."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1  # cosine decays lr -> lr * min_lr_ratio over max_steps

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerConfig:
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        if "betas" in kwargs and isinstance(kwargs["betas"], list):
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs)


@dataclass
class LoopConfig:
    """How long to run, how often to log, checkpoint, and sample."""

    max_steps: int = 1000
    log_every: int = 10
    seed: int = 0
    ckpt_every: int = 0  # 0 disables checkpointing
    ckpt_dir: str = "artifacts/checkpoints"
    ckpt_keep_last: int = 3  # keep only the N most recent step checkpoints
    # "auto" = resume from newest ckpt in ckpt_dir if present, else start fresh;
    # "never" = always start fresh even if checkpoints exist.
    # CLI --fresh / --resume / --resume-latest override this value.
    resume: Literal["auto", "never"] = "auto"
    # Qualitative sampling during training. 0 disables; anything else requires
    # sample_prompts to be non-empty.
    sample_every: int = 0
    sample_prompts: tuple[str, ...] = ()
    sample_new_tokens: int = 5
    sample_temperature: float = 1.0
    sample_top_k: int | None = 50

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.ckpt_every < 0:
            raise ValueError(f"ckpt_every must be >= 0, got {self.ckpt_every}")
        if self.ckpt_keep_last < 1:
            raise ValueError(f"ckpt_keep_last must be >= 1, got {self.ckpt_keep_last}")
        if self.sample_every < 0:
            raise ValueError(f"sample_every must be >= 0, got {self.sample_every}")
        if self.sample_every > 0:
            if not self.sample_prompts:
                raise ValueError("sample_every > 0 requires non-empty sample_prompts")
            if self.sample_new_tokens < 1:
                raise ValueError(
                    f"sample_new_tokens must be >= 1, got {self.sample_new_tokens}"
                )
        if self.sample_temperature <= 0:
            raise ValueError(
                f"sample_temperature must be > 0, got {self.sample_temperature}"
            )
        if self.sample_top_k is not None and self.sample_top_k < 1:
            raise ValueError(
                f"sample_top_k must be None or >= 1, got {self.sample_top_k}"
            )
        if self.resume not in ("auto", "never"):
            raise ValueError(f"resume must be 'auto' or 'never', got {self.resume!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopConfig:
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        if "sample_prompts" in kwargs and isinstance(kwargs["sample_prompts"], list):
            kwargs["sample_prompts"] = tuple(kwargs["sample_prompts"])
        return cls(**kwargs)


@dataclass
class TrainStepConfig:
    """Hyperparameters for :func:`training.trainer.train_step`.

    Maps to the previous ``alpha`` / ``beta`` naming as
    ``load_balance_weight`` / ``router_z_weight``.
    """

    load_balance_weight: float = 0.01
    router_z_weight: float = 0.001
    grad_clip: float = 1.0
    device: str = "cuda"
    autocast_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    use_autocast: bool = True

    def __post_init__(self) -> None:
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.load_balance_weight < 0 or self.router_z_weight < 0:
            raise ValueError("MoE loss weights must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainStepConfig:
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, section: str | None = "train_step"
    ) -> TrainStepConfig:
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"YAML file is empty: {path}")
        if section is not None:
            if not isinstance(raw, dict) or section not in raw:
                raise KeyError(f"Missing key {section!r} in {path}")
            raw = raw[section]
        elif not isinstance(raw, dict):
            raise TypeError(f"YAML root must be a mapping when section is None: {path}")
        if not isinstance(raw, dict):
            raise TypeError(
                f"train_step YAML must map to an object, got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

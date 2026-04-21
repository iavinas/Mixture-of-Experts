"""Construct ``MoELM`` from ``MoELMConfig`` (including YAML on disk)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import MoELMConfig
from .moe_lm import MoELM


def build_moe_lm(config: MoELMConfig) -> MoELM:
    """Construct ``MoELM`` from a config object."""
    return MoELM(**asdict(config))


def build_moe_lm_from_yaml(path: str | Path, *, section: str | None = None) -> MoELM:
    """Load ``MoELMConfig`` from YAML and build the model."""
    return build_moe_lm(MoELMConfig.from_yaml(path, section=section))

"""YAML-driven settings for streaming LM datasets and PyTorch loaders."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StreamingPackedDataConfig:
    """HF text dataset packed into (input, target) chunks of length ``seq_len``.

    If ``streaming`` is False (default), ``datasets`` downloads and caches the split
    locally (Hub cache), then iterates from disk—good for offline training after the
    first download. If True, examples are fetched incrementally without full download.
    """

    dataset_name: str
    split: str = "train"
    text_column: str = "text"
    tokenizer_name_or_path: str = "gpt2"
    seq_len: int = 512
    end_of_text_token: str = "<|endoftext|>"
    trust_remote_code: bool = False
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {self.seq_len}")
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamingPackedDataConfig:
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, section: str | None = None
    ) -> StreamingPackedDataConfig:
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


@dataclass
class DataLoaderConfig:
    """Arguments for :class:`torch.utils.data.DataLoader` over iterable datasets."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}"
            )
        if self.persistent_workers and self.num_workers == 0:
            # PyTorch raises at runtime — fail fast with a clearer message.
            raise ValueError("persistent_workers requires num_workers >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataLoaderConfig:
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, section: str | None = None
    ) -> DataLoaderConfig:
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

"""Build tokenizers and DataLoaders from config objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from .config import DataLoaderConfig, StreamingPackedDataConfig
from .packed_streaming import PackedTinyStories


def load_tokenizer(name_or_path: str) -> Tokenizer:
    """Load a :class:`tokenizers.Tokenizer` (e.g. ``gpt2`` from the Hub)."""
    return Tokenizer.from_pretrained(name_or_path)


def build_packed_lm_dataloader(
    data_cfg: StreamingPackedDataConfig,
    loader_cfg: DataLoaderConfig,
) -> DataLoader:
    """Packed LM :class:`~torch.utils.data.DataLoader` (Hub stream or cached split)."""
    tok = load_tokenizer(data_cfg.tokenizer_name_or_path)
    ds = PackedTinyStories(
        split=data_cfg.split,
        tokenizer=tok,
        seq_len=data_cfg.seq_len,
        dataset_name=data_cfg.dataset_name,
        text_column=data_cfg.text_column,
        end_of_text_token=data_cfg.end_of_text_token,
        trust_remote_code=data_cfg.trust_remote_code,
        streaming=data_cfg.streaming,
    )
    kwargs: dict = {
        "batch_size": loader_cfg.batch_size,
        "num_workers": loader_cfg.num_workers,
        "pin_memory": loader_cfg.pin_memory,
        "drop_last": loader_cfg.drop_last,
    }
    if loader_cfg.num_workers > 0:
        kwargs["prefetch_factor"] = loader_cfg.prefetch_factor
        kwargs["persistent_workers"] = loader_cfg.persistent_workers
    return DataLoader(ds, **kwargs)


def build_packed_lm_dataloader_from_yaml(
    data_yaml: str | Path,
    loader_yaml: str | Path,
    *,
    data_section: str | None = None,
    loader_section: str | None = None,
) -> DataLoader:
    """Convenience: load two YAML files and build the loader."""
    data_cfg = StreamingPackedDataConfig.from_yaml(data_yaml, section=data_section)
    loader_cfg = DataLoaderConfig.from_yaml(loader_yaml, section=loader_section)
    return build_packed_lm_dataloader(data_cfg, loader_cfg)


def build_packed_lm_dataloader_from_unified_yaml(
    path: str | Path,
    *,
    data_section: str = "data",
    loader_section: str = "dataloader",
) -> DataLoader:
    """Single experiment YAML with ``data`` and ``dataloader`` mappings."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping (path={path})")
    if data_section not in raw or loader_section not in raw:
        raise KeyError(
            f"Expected keys {data_section!r} and {loader_section!r} in {path}"
        )
    data_cfg = StreamingPackedDataConfig.from_dict(raw[data_section])
    loader_cfg = DataLoaderConfig.from_dict(raw[loader_section])
    return build_packed_lm_dataloader(data_cfg, loader_cfg)

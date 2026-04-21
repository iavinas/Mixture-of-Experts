"""Stream a HF dataset, tokenize, and yield fixed-length LM chunks."""

from __future__ import annotations

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset


class PackedTinyStories(IterableDataset):
    """Tokenize a HF text split and yield ``(input_ids, target_ids)`` LM chunks.

    ``target_ids`` is ``input_ids`` shifted by one. Works with any HF dataset that
    exposes a text column; defaults match TinyStories.

    Use ``streaming=True`` for incremental Hub reads; ``False`` to download/cache
    the split (see :class:`StreamingPackedDataConfig.streaming`).
    """

    def __init__(
        self,
        *,
        split: str,
        tokenizer: Tokenizer,
        seq_len: int,
        dataset_name: str,
        text_column: str = "text",
        end_of_text_token: str = "<|endoftext|>",
        trust_remote_code: bool = False,
        streaming: bool = False,
    ):
        self.split = split
        self.tok = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.trust_remote_code = trust_remote_code
        self.streaming = streaming
        tid = tokenizer.token_to_id(end_of_text_token)
        if tid is None:
            raise ValueError(
                f"Tokenizer has no id for end_of_text_token={end_of_text_token!r}"
            )
        self.eot = tid

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )
        if (
            worker_info is not None
            and worker_info.num_workers > 0
            and hasattr(ds, "shard")
        ):
            ds = ds.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        buf: list[int] = []
        for ex in ds:
            ids = self.tok.encode(ex[self.text_column]).ids + [self.eot]
            buf.extend(ids)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
                buf = buf[self.seq_len :]

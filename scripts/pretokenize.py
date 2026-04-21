"""Pre-tokenize a HF text dataset into a single flat uint32 .bin file.

Use this before expensive GPU runs so training I/O is a contiguous mmap read
instead of per-epoch tokenize+stream. The resulting file is a raw stream of
token ids with ``end_of_text_token`` between documents.

Example::

    PYTHONPATH=src python scripts/pretokenize.py \\
        --dataset roneneldan/TinyStories --split train \\
        --tokenizer gpt2 --out data/processed/tinystories_train.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize a HF text dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--eot", default="<|endoftext|>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward to datasets.load_dataset",
    )
    args = parser.parse_args()

    tok = Tokenizer.from_pretrained(args.tokenizer)
    eot_id = tok.token_to_id(args.eot)
    if eot_id is None:
        raise ValueError(f"Tokenizer has no id for {args.eot!r}")

    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=False,
        trust_remote_code=args.trust_remote_code,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("wb") as f:
        for ex in ds:
            ids = tok.encode(ex[args.text_column]).ids
            ids.append(eot_id)
            arr = np.asarray(ids, dtype=np.uint32)
            f.write(arr.tobytes())
            total += arr.size
    print(f"wrote {total:,} tokens to {args.out}", flush=True)


if __name__ == "__main__":
    main()

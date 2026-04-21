"""Train MoELM from one experiment YAML (data, model, train_step, optimizer, loop).

Run from repo root (see README for PYTHONPATH)::

    python scripts/train.py --config configs/experiments/toy_debug_train.yaml

Quick smoke::

    python scripts/train.py --config <path> --max-steps 5

Needs network on first run if ``data.streaming`` is false (dataset is cached
after download) or always if ``streaming`` is true.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import warnings
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import yaml
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.config import DataLoaderConfig, StreamingPackedDataConfig
from data.loaders import build_packed_lm_dataloader, load_tokenizer
from moe.models.config import MoELMConfig
from moe.models.moe_stack import build_moe_lm
from training.config import LoopConfig, OptimizerConfig, TrainStepConfig
from training.loops import (
    basic_training_loop,
    find_latest_checkpoint,
    load_checkpoint,
)


def load_experiment(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Experiment YAML must be a mapping: {path}")
    required = ("data", "dataloader", "model", "train_step", "optimizer", "training")
    for key in required:
        if key not in raw:
            raise KeyError(f"Experiment YAML missing {key!r}: {path}")
    return raw


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_lr_lambda(opt_cfg: OptimizerConfig, max_steps: int):
    """Linear warmup → cosine decay to ``lr * min_lr_ratio`` over ``max_steps``."""
    warmup = opt_cfg.warmup_steps
    min_ratio = opt_cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if max_steps <= warmup:
            return 1.0
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic MoELM training loop")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Unified YAML (data, dataloader, model, train_step, optimizer, training)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps for quick debugging",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a specific checkpoint file (.pt).",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the newest checkpoint in training.ckpt_dir, if any.",
    )
    args = parser.parse_args()

    exp = load_experiment(args.config.resolve())

    loop_cfg = LoopConfig.from_dict(exp["training"])
    if args.max_steps is not None:
        loop_cfg = replace(loop_cfg, max_steps=args.max_steps)
    _seed_everything(loop_cfg.seed)

    data_cfg = StreamingPackedDataConfig.from_dict(exp["data"])
    loader_cfg = DataLoaderConfig.from_dict(exp["dataloader"])
    dataloader = build_packed_lm_dataloader(data_cfg, loader_cfg)

    tokenizer = load_tokenizer(data_cfg.tokenizer_name_or_path)
    vocab_size = tokenizer.get_vocab_size()

    model_raw = dict(exp["model"])
    model_raw["vocab_size"] = vocab_size
    # RoPE / forward require max_seq_len >= packed chunk length (data.seq_len).
    # Only vocab_size was synced before; align this so one YAML cannot drift.
    seq_need = data_cfg.seq_len
    cur_max = model_raw.get("max_seq_len")
    if cur_max is None:
        model_raw["max_seq_len"] = seq_need
    elif int(cur_max) < seq_need:
        warnings.warn(
            f"model.max_seq_len ({cur_max}) < data.seq_len ({seq_need}); "
            f"raising max_seq_len to {seq_need} so training batches fit.",
            UserWarning,
            stacklevel=1,
        )
        model_raw["max_seq_len"] = seq_need
    model_cfg = MoELMConfig.from_dict(model_raw)
    model = build_moe_lm(model_cfg)

    train_cfg = TrainStepConfig.from_dict(exp["train_step"])
    opt_cfg = OptimizerConfig.from_dict(exp["optimizer"])

    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=opt_cfg.betas,
    )
    scheduler = LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(opt_cfg, loop_cfg.max_steps)
    )

    # GradScaler is only meaningful for fp16 autocast on CUDA.
    # bf16 (CUDA) and cpu/mps do not need loss scaling; an enabled scaler there
    # can introduce spurious infs, so keep it disabled.
    device = torch.device(train_cfg.device)
    scaler_enabled = (
        device.type == "cuda"
        and train_cfg.use_autocast
        and train_cfg.autocast_dtype == "float16"
    )
    grad_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(grad_device, enabled=scaler_enabled)

    # Resolve resume source: explicit path wins, then --resume-latest scans
    # the configured ckpt_dir. Load before training so optimizer/scheduler
    # state is in place on the target device.
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = args.resume
    elif args.resume_latest:
        resume_path = find_latest_checkpoint(loop_cfg.ckpt_dir)
        if resume_path is None:
            print(
                f"[ckpt] --resume-latest: no checkpoints in {loop_cfg.ckpt_dir}, "
                f"starting from scratch.",
                flush=True,
            )

    start_step = 0
    if resume_path is not None:
        model.to(device)
        start_step = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device=device
        )

    basic_training_loop(
        model,
        dataloader,
        optimizer,
        scaler,
        scheduler,
        train_cfg,
        loop_cfg,
        tokenizer=tokenizer,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()

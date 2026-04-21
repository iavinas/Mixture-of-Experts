"""Standalone MoELM evaluation: CE / perplexity / acc@1 + per-MoE-layer expert usage.

Loads the latest checkpoint from ``training.ckpt_dir`` in the experiment YAML,
runs inference on ``--max-batches`` batches of the requested split, and prints
the metrics. Does not touch training code.

    PYTHONPATH=src python scripts/evaluate.py \\
        --config configs/experiments/gpu_train.yaml \\
        --split validation --max-batches 50
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn.functional as F
import yaml

from data.config import DataLoaderConfig, StreamingPackedDataConfig
from data.loaders import build_packed_lm_dataloader, load_tokenizer
from moe.models.config import MoELMConfig
from moe.models.moe_stack import build_moe_lm
from training.config import LoopConfig, TrainStepConfig
from training.loops import find_latest_checkpoint
from training.trainer import build_autocast_ctx


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MoELM checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override train_step.device (e.g. 'mps' or 'cpu' on a Mac).",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help="Override train_step.autocast_dtype (pair with --device when moving "
        "a CUDA/bf16 config onto mps/cpu).",
    )
    args = parser.parse_args()

    with args.config.resolve().open(encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    # Override split so we evaluate held-out data, not the training split.
    data_raw = dict(exp["data"])
    data_raw["split"] = args.split
    data_cfg = StreamingPackedDataConfig.from_dict(data_raw)
    loader_cfg = DataLoaderConfig.from_dict(exp["dataloader"])
    dataloader = build_packed_lm_dataloader(data_cfg, loader_cfg)

    tokenizer = load_tokenizer(data_cfg.tokenizer_name_or_path)
    model_raw = dict(exp["model"])
    model_raw["vocab_size"] = tokenizer.get_vocab_size()
    if int(model_raw.get("max_seq_len", 0)) < data_cfg.seq_len:
        model_raw["max_seq_len"] = data_cfg.seq_len
    model = build_moe_lm(MoELMConfig.from_dict(model_raw))

    train_cfg = TrainStepConfig.from_dict(exp["train_step"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    if args.autocast_dtype is not None:
        train_cfg = replace(train_cfg, autocast_dtype=args.autocast_dtype)
    loop_cfg = LoopConfig.from_dict(exp["training"])
    device = torch.device(train_cfg.device)
    model.to(device)

    ckpt_path = find_latest_checkpoint(loop_cfg.ckpt_dir)
    if ckpt_path is None:
        raise SystemExit(f"no checkpoints found in {loop_cfg.ckpt_dir}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    step = int(blob.get("step", 0))
    print(
        f"=== Eval on {data_cfg.dataset_name}[{args.split}], "
        f"step={step}, up to {args.max_batches} batches ===",
        flush=True,
    )
    print(f"[ckpt] {ckpt_path}", flush=True)

    model.eval()
    n_exp = model.n_experts
    loss_sum = 0.0  # sum of per-token CE (nats)
    token_count = 0
    correct1 = 0
    usage_counts: dict[int, torch.Tensor] = {}  # moe-layer-idx -> bincount over experts

    amp_ctx = build_autocast_ctx(device, train_cfg)
    seen = 0
    with torch.no_grad(), amp_ctx:
        for batch in dataloader:
            if seen >= args.max_batches:
                break
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits, _, routing = model(inputs)
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_targets = targets.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
            loss_sum += loss.item()
            token_count += flat_targets.numel()
            correct1 += (flat_logits.argmax(dim=-1) == flat_targets).sum().item()

            for i, (_, _, topk_idx) in enumerate(routing):
                counts = torch.bincount(topk_idx.flatten(), minlength=n_exp)
                usage_counts[i] = (
                    counts if i not in usage_counts else usage_counts[i] + counts
                )
            seen += 1

    if token_count == 0:
        raise SystemExit("no tokens evaluated — split may be empty or max-batches=0")

    ce = loss_sum / token_count
    ppl = math.exp(ce)
    acc1 = correct1 / token_count

    print(f"tokens_evaluated  {token_count:,}")
    print(f"cross_entropy     {ce:.4f} nats/token")
    print(f"perplexity        {ppl:.2f}")
    print(f"acc@1             {acc1 * 100:.2f}%")

    if usage_counts:
        print("\nRouting (per MoE layer, % of top-k slots per expert)")
        for i in sorted(usage_counts):
            counts = usage_counts[i].float()
            pct = 100.0 * counts / counts.sum().clamp_min(1.0)
            row = "  ".join(f"{v:5.1f}" for v in pct.tolist())
            print(f"  moe_L{i}  [{row}]")


if __name__ == "__main__":
    main()

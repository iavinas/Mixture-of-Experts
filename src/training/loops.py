"""Minimal training loop with EMA logging, NaN guard, and checkpointing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from moe.models.moe_lm import MoELM

from .config import LoopConfig, TrainStepConfig
from .trainer import build_autocast_ctx, train_step


class _TokenizerLike(Protocol):
    """Minimum tokenizer surface the sampler uses (matches tokenizers.Tokenizer)."""

    def encode(self, text: str): ...  # returns an object exposing `.ids`
    def decode(self, ids: list[int]) -> str: ...


def basic_training_loop(
    model: MoELM,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: LRScheduler,
    train_cfg: TrainStepConfig,
    loop_cfg: LoopConfig,
    *,
    tokenizer: _TokenizerLike | None = None,
) -> None:
    """Run ``train_step`` until ``loop_cfg.max_steps``.

    Streaming loaders may never exhaust, so re-``iter`` on ``StopIteration``.
    Logs an EMA of ``lm``/``total`` and checkpoints every ``loop_cfg.ckpt_every``
    steps (0 disables). A non-finite loss at a log step raises.

    If ``loop_cfg.sample_every > 0``, a ``tokenizer`` must be supplied so the
    loop can generate text from ``loop_cfg.sample_prompts`` at that cadence.
    """
    device = torch.device(train_cfg.device)
    model.train()
    model.to(device)

    if loop_cfg.sample_every > 0 and tokenizer is None:
        raise RuntimeError(
            "sample_every > 0 but no tokenizer was passed to basic_training_loop"
        )

    _print_banner(model, device, train_cfg, loop_cfg)

    ckpt_dir = Path(loop_cfg.ckpt_dir)
    if loop_cfg.ckpt_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    ema_lm: float | None = None
    ema_total: float | None = None
    alpha = 0.1  # EMA smoothing factor

    step = 0
    data_iter = iter(dataloader)

    while step < loop_cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step += 1
        is_log_step = step % loop_cfg.log_every == 0 or step == loop_cfg.max_steps
        metrics = train_step(
            model,
            batch,
            optimizer,
            scaler,
            scheduler,
            train_cfg,
            collect_scalars=is_log_step,
        )

        if is_log_step:
            lm = metrics["lm"]
            total = metrics["total"]
            if not (math.isfinite(lm) and math.isfinite(total)):
                raise RuntimeError(
                    f"Non-finite loss at step {step}: lm={lm}, total={total}. "
                    f"Grads likely diverged — reduce lr or check data."
                )
            ema_lm = lm if ema_lm is None else (1 - alpha) * ema_lm + alpha * lm
            ema_total = (
                total if ema_total is None else (1 - alpha) * ema_total + alpha * total
            )
            _log_step(step, loop_cfg.max_steps, metrics, ema_lm, ema_total, optimizer)

        if loop_cfg.ckpt_every > 0 and step % loop_cfg.ckpt_every == 0:
            _save_checkpoint(
                ckpt_dir, step, model, optimizer, scheduler, scaler, loop_cfg
            )

        if loop_cfg.sample_every > 0 and step % loop_cfg.sample_every == 0:
            _sample_and_log(model, tokenizer, device, train_cfg, loop_cfg, step)


def _log_step(
    step: int,
    max_steps: int,
    metrics: dict[str, Any],
    ema_lm: float,
    ema_total: float,
    optimizer: Optimizer,
) -> None:
    lm = metrics["lm"]
    total = metrics["total"]
    aux = metrics["aux"]
    aux_n = metrics["aux_norm"]
    z = metrics["z"]
    lr = optimizer.param_groups[0]["lr"]
    print(
        f"step {step}/{max_steps}  lr={lr:.2e}  "
        f"lm={lm:.4f} (ema {ema_lm:.4f})  "
        f"aux={aux:.4f}  aux_n={aux_n:.4f}  "
        f"z={z:.4f}  total={total:.4f} (ema {ema_total:.4f})",
        flush=True,
    )


def _save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: MoELM,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    loop_cfg: LoopConfig,
) -> None:
    path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    # Keep only the most recent `ckpt_keep_last` checkpoints to bound disk use.
    existing = sorted(ckpt_dir.glob("step_*.pt"))
    for old in existing[: -loop_cfg.ckpt_keep_last]:
        try:
            old.unlink()
        except OSError:
            pass
    print(f"[ckpt] saved {path}", flush=True)


def _sample_and_log(
    model: MoELM,
    tokenizer: _TokenizerLike,
    device: torch.device,
    train_cfg: TrainStepConfig,
    loop_cfg: LoopConfig,
    step: int,
) -> None:
    """Generate a short continuation for each prompt and print it.

    Restores ``model.train()`` afterwards so the next step is not silently in
    eval mode (``MoELM.generate`` only sets eval, never resets).
    """
    was_training = model.training
    model.eval()
    try:
        amp_ctx = build_autocast_ctx(device, train_cfg)
        with torch.no_grad(), amp_ctx:
            for prompt_text in loop_cfg.sample_prompts:
                ids = tokenizer.encode(prompt_text).ids
                if not ids:
                    continue
                prompt = torch.tensor([ids], dtype=torch.long, device=device)
                # MoELM enforces past_len + T <= max_seq_len; left-truncate the
                # prompt so there is room for sample_new_tokens.
                cap = model.max_seq_len - loop_cfg.sample_new_tokens
                if prompt.size(1) > cap:
                    prompt = prompt[:, -max(cap, 1) :]
                out = model.generate(
                    prompt,
                    max_new_tokens=loop_cfg.sample_new_tokens,
                    temperature=loop_cfg.sample_temperature,
                    top_k=loop_cfg.sample_top_k,
                )
                new_ids = out[0, prompt.size(1) :].tolist()
                cont = tokenizer.decode(new_ids)
                print(
                    f"[sample step {step}] prompt={prompt_text!r} cont={cont!r}",
                    flush=True,
                )
    finally:
        if was_training:
            model.train()


def _print_banner(
    model: MoELM,
    device: torch.device,
    train_cfg: TrainStepConfig,
    loop_cfg: LoopConfig,
) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train] device={device}  autocast={train_cfg.use_autocast}/"
        f"{train_cfg.autocast_dtype}  params={n_params:,}  "
        f"trainable={n_trainable:,}  max_steps={loop_cfg.max_steps}  "
        f"seed={loop_cfg.seed}  ckpt_every={loop_cfg.ckpt_every}",
        flush=True,
    )
    print(
        "[train] load_balance_loss is aux_n=1.0 at perfect uniform routing "
        "(minimum is k per layer, we normalize by n_moe_layers * k).",
        flush=True,
    )
    if loop_cfg.sample_every > 0:
        print(
            f"[train] sampling every {loop_cfg.sample_every} steps: "
            f"{len(loop_cfg.sample_prompts)} prompts × "
            f"{loop_cfg.sample_new_tokens} tokens, "
            f"temp={loop_cfg.sample_temperature} top_k={loop_cfg.sample_top_k}",
            flush=True,
        )

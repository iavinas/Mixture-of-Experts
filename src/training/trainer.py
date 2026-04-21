"""Minimal training step for MoELM (LM + load-balance + router z-loss)."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from moe.losses import load_balance_loss, router_z_loss
from moe.models.moe_lm import MoELM

from .config import TrainStepConfig


def build_autocast_ctx(device: torch.device, cfg: TrainStepConfig) -> Any:
    """Return the autocast context manager to use for forward/generate.

    Centralizes the device+dtype rules so training and sampling stay consistent:
    MPS+fp32 uses a no-op (MPS autocast rejects fp32), CPU+fp16 disables
    autocast (unsupported), everything else honors ``cfg.use_autocast``.
    """
    if device.type == "cuda":
        amp_device = "cuda"
    elif device.type == "mps":
        amp_device = "mps"
    else:
        amp_device = "cpu"
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.autocast_dtype]
    use_autocast = cfg.use_autocast and cfg.autocast_dtype != "float32"
    if amp_device == "cpu" and cfg.autocast_dtype == "float16":
        use_autocast = False
    if not use_autocast:
        return nullcontext()
    return torch.autocast(device_type=amp_device, dtype=amp_dtype)


def train_step(
    model: MoELM,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: LRScheduler,
    cfg: TrainStepConfig,
    *,
    collect_scalars: bool = False,
) -> dict[str, Any]:
    """One training step: forward, LM + MoE aux losses, backward, optimizer step.

    Returns a dict with on-device tensors by default; set ``collect_scalars=True``
    (on log steps only) to pull floats to the host — otherwise we avoid the
    per-step GPU→CPU sync that ``.item()`` / ``.cpu()`` would force.
    """
    inputs, targets = batch
    device = torch.device(cfg.device)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with build_autocast_ctx(device, cfg):
        logits, _, routing = model(inputs)

        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        n_exp = model.n_experts
        if routing:
            # stack(...).sum() avoids Python int-start scalar add per layer.
            aux = torch.stack(
                [load_balance_loss(p, idx, n_exp) for (p, _, idx) in routing]
            ).sum()
            zl = torch.stack([router_z_loss(lg) for (_, lg, _) in routing]).sum()
            # Per-layer minimum is k (top-k slots per token); sum over MoE layers.
            k_slots = routing[0][2].shape[1]
            aux_norm = aux / (len(routing) * k_slots)
        else:
            aux = torch.zeros((), device=logits.device, dtype=lm_loss.dtype)
            zl = torch.zeros((), device=logits.device, dtype=lm_loss.dtype)
            aux_norm = aux

        total = lm_loss + cfg.load_balance_weight * aux + cfg.router_z_weight * zl

    scaler.scale(total).backward()
    # `unscale_` is a no-op on a disabled scaler; grad-clip still runs on raw grads.
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scale_before = scaler.get_scale() if scaler.is_enabled() else None
    scaler.step(optimizer)
    scaler.update()
    # When the scaler is enabled (CUDA+fp16), a skipped step is signalled by the
    # scale shrinking on the next `update`. In every other case the step always
    # took effect and the scheduler should advance.
    if scale_before is None:
        stepped = True
    else:
        stepped = scaler.get_scale() >= scale_before
    if stepped:
        scheduler.step()

    result: dict[str, Any] = {
        "lm": lm_loss.detach(),
        "aux": aux.detach(),
        "aux_norm": aux_norm.detach(),
        "z": zl.detach(),
        "total": total.detach(),
        "stepped": stepped,
    }
    if collect_scalars:
        with torch.no_grad():
            if routing:
                last_idx = routing[-1][2]
                usage = torch.bincount(last_idx.flatten(), minlength=n_exp).float()
                usage = usage / usage.sum().clamp_min(1.0)
            else:
                usage = torch.zeros(n_exp, device=logits.device)
        result = {
            "lm": lm_loss.item(),
            "aux": aux.item(),
            "aux_norm": aux_norm.item(),
            "z": zl.item(),
            "total": total.item(),
            "usage": usage.cpu(),
            "stepped": stepped,
        }
    return result


def load_train_step_config(
    path: str | Path, *, section: str | None = "train_step"
) -> TrainStepConfig:
    """Load :class:`TrainStepConfig` from YAML (defaults to ``train_step:`` section)."""
    return TrainStepConfig.from_yaml(path, section=section)

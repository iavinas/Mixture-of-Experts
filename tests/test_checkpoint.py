"""Checkpoint save → resume round-trip on a tiny MoELM."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from moe.models.config import MoELMConfig
from moe.models.moe_stack import build_moe_lm
from training.config import LoopConfig, TrainStepConfig
from training.loops import (
    basic_training_loop,
    find_latest_checkpoint,
    load_checkpoint,
)


class _FakeDS(IterableDataset):
    """Deterministic packed (input, target) generator."""

    def __init__(self, vocab: int, seq_len: int):
        self.v, self.s = vocab, seq_len

    def __iter__(self):
        g = torch.Generator().manual_seed(0)
        while True:
            ids = torch.randint(0, self.v, (self.s + 1,), generator=g)
            yield ids[:-1].long(), ids[1:].long()


def _fresh_run(tmp_path: Path, max_steps: int, start_step: int = 0) -> dict:
    cfg = MoELMConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        n_experts=4,
        k=2,
        n_shared=1,
        max_seq_len=16,
    )
    model = build_moe_lm(cfg)
    dl = DataLoader(_FakeDS(64, 8), batch_size=2)
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = LambdaLR(opt, lr_lambda=lambda step: 1.0)
    scaler = GradScaler("cpu", enabled=False)
    train_cfg = TrainStepConfig(
        device="cpu", autocast_dtype="float32", use_autocast=False
    )
    loop_cfg = LoopConfig(
        max_steps=max_steps,
        log_every=1000,  # keep output quiet
        ckpt_every=2,
        ckpt_dir=str(tmp_path),
        ckpt_keep_last=3,
    )
    basic_training_loop(
        model, dl, opt, scaler, sched, train_cfg, loop_cfg, start_step=start_step
    )
    return {"model": model, "opt": opt, "sched": sched, "scaler": scaler}


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    _fresh_run(tmp_path, max_steps=4)  # writes step_00000002.pt, step_00000004.pt
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "step_00000004.pt"

    # Fresh scaffolding, then load from the checkpoint.
    torch.manual_seed(123)  # different init seed on purpose
    cfg = MoELMConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        n_experts=4,
        k=2,
        n_shared=1,
        max_seq_len=16,
    )
    m2 = build_moe_lm(cfg)
    opt2 = AdamW(m2.parameters(), lr=1e-3)
    sched2 = LambdaLR(opt2, lr_lambda=lambda step: 1.0)
    scaler2 = GradScaler("cpu", enabled=False)

    step = load_checkpoint(latest, m2, opt2, sched2, scaler2, device="cpu")
    assert step == 4

    # State should match the training-run state at that step.
    blob = torch.load(latest, map_location="cpu", weights_only=False)
    for k, v in blob["model"].items():
        assert torch.equal(m2.state_dict()[k], v), f"mismatch on {k}"


def test_start_step_clamp_noop_when_done(tmp_path: Path) -> None:
    # max_steps=1, start_step=1 → loop does nothing, no checkpoint files.
    cfg = MoELMConfig(
        vocab_size=64, d_model=32, n_heads=4, d_ff=64, max_seq_len=8, n_layers=2
    )
    m = build_moe_lm(cfg)
    dl = DataLoader(_FakeDS(64, 4), batch_size=2)
    opt = AdamW(m.parameters(), lr=1e-3)
    sched = LambdaLR(opt, lr_lambda=lambda step: 1.0)
    scaler = GradScaler("cpu", enabled=False)
    train_cfg = TrainStepConfig(
        device="cpu", autocast_dtype="float32", use_autocast=False
    )
    loop_cfg = LoopConfig(
        max_steps=1, log_every=1, ckpt_every=1, ckpt_dir=str(tmp_path)
    )
    basic_training_loop(m, dl, opt, scaler, sched, train_cfg, loop_cfg, start_step=1)
    assert list(tmp_path.glob("step_*.pt")) == []


def test_find_latest_missing_dir(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None
    assert find_latest_checkpoint(tmp_path) is None  # exists but empty

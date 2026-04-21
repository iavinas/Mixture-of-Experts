"""Smoke-test the periodic sampling hook on a tiny MoELM with a stub tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe.models.config import MoELMConfig
from moe.models.moe_stack import build_moe_lm
from training.config import LoopConfig, TrainStepConfig
from training.loops import _sample_and_log


@dataclass
class _Encoded:
    ids: list[int]


class _StubTokenizer:
    """Minimum surface `_sample_and_log` needs — `.encode(...).ids` + `.decode(...)`."""

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.last_decoded: list[int] | None = None

    def encode(self, text: str) -> _Encoded:
        # Map each character to an id mod vocab so the test is deterministic.
        return _Encoded(ids=[(ord(c) % self.vocab) or 1 for c in text][:8])

    def decode(self, ids: list[int]) -> str:
        self.last_decoded = list(ids)
        return "<decoded:" + ",".join(str(i) for i in ids) + ">"


def _tiny_model(max_seq_len: int = 16) -> torch.nn.Module:
    cfg = MoELMConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        n_experts=4,
        k=2,
        n_shared=1,
        max_seq_len=max_seq_len,
    )
    return build_moe_lm(cfg)


def test_sample_and_log_restores_training_mode(capsys) -> None:
    model = _tiny_model()
    model.train()
    tok = _StubTokenizer(vocab=64)
    loop_cfg = LoopConfig(
        max_steps=10,
        sample_every=5,
        sample_prompts=("hello", "world"),
        sample_new_tokens=3,
        sample_top_k=None,
    )
    train_cfg = TrainStepConfig(
        device="cpu", autocast_dtype="float32", use_autocast=False
    )

    _sample_and_log(model, tok, torch.device("cpu"), train_cfg, loop_cfg, step=5)

    # Restored to training mode.
    assert model.training is True
    # Decoded exactly sample_new_tokens per prompt (last call's ids).
    assert tok.last_decoded is not None
    assert len(tok.last_decoded) == 3

    out = capsys.readouterr().out
    assert "[sample step 5]" in out
    assert "prompt='hello'" in out
    assert "prompt='world'" in out


def test_sample_and_log_left_truncates_long_prompt(capsys) -> None:
    # max_seq_len=8, ask for 3 new tokens → cap = 5, prompt (8 chars) must truncate.
    model = _tiny_model(max_seq_len=8)
    tok = _StubTokenizer(vocab=64)
    loop_cfg = LoopConfig(
        max_steps=10,
        sample_every=5,
        sample_prompts=("abcdefgh",),
        sample_new_tokens=3,
        sample_top_k=None,
    )
    train_cfg = TrainStepConfig(
        device="cpu", autocast_dtype="float32", use_autocast=False
    )

    _sample_and_log(model, tok, torch.device("cpu"), train_cfg, loop_cfg, step=5)

    # Does not raise, prints a sample line.
    out = capsys.readouterr().out
    assert "[sample step 5]" in out


def test_loop_rejects_missing_tokenizer_when_sampling_enabled() -> None:
    from torch.amp import GradScaler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader, IterableDataset

    from moe.models.config import MoELMConfig
    from moe.models.moe_stack import build_moe_lm
    from training.loops import basic_training_loop

    class _FakeDS(IterableDataset):
        def __iter__(self):
            while True:
                yield torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)

    model = build_moe_lm(
        MoELMConfig(vocab_size=64, d_model=32, n_heads=4, d_ff=64, max_seq_len=8)
    )
    dl = DataLoader(_FakeDS(), batch_size=2)
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = LambdaLR(opt, lr_lambda=lambda _: 1.0)
    scaler = GradScaler("cpu", enabled=False)
    train_cfg = TrainStepConfig(
        device="cpu", autocast_dtype="float32", use_autocast=False
    )
    loop_cfg = LoopConfig(
        max_steps=2,
        sample_every=1,
        sample_prompts=("hi",),
    )

    try:
        basic_training_loop(model, dl, opt, scaler, sched, train_cfg, loop_cfg)
    except RuntimeError as e:
        assert "sample_every" in str(e)
    else:
        raise AssertionError("expected RuntimeError when tokenizer is missing")

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Training always runs with `src/` on `PYTHONPATH` — use the Makefile, which sets this for you:

```bash
make train                                  # uses configs/experiments/toy_debug_train.yaml
make train TRAIN_CONFIG=configs/experiments/toy_train_data.yaml
make train TRAIN_ARGS='--max-steps 20'      # override training.max_steps for a smoke run
```

Running the script directly requires the env prefix:

```bash
PYTHONPATH=src python scripts/train.py --config <experiment.yaml> [--max-steps N]
```

Lint / format (ruff, configured via `ruff.toml`):

```bash
make lint          # ruff check src tests scripts
make format        # ruff format (writes)
make format-check  # ruff format --check (CI-style)
make fix           # ruff check --fix
```

Tests use pytest with `pythonpath = ["src"]` from `pyproject.toml`, so `pytest` alone works from the repo root:

```bash
pytest                                      # full suite
pytest tests/test_smoke.py::test_smoke      # single test
```

First training run needs network: HF downloads the TinyStories dataset (cached on disk when `data.streaming: false`) and the GPT-2 tokenizer. With `streaming: true`, network is required every run.

## Architecture

Three top-level packages under `src/` (no `__init__.py` at `src/` itself; `PYTHONPATH=src` makes them importable as top-level `data`, `moe`, `training`).

### `moe/` — the model

`MoELM` (`moe/models/moe_lm.py`) is a pre-norm transformer LM with weight-tied embeddings and a **single shared `RotaryEmbedding`** instance passed into every block (RoPE tables live as buffers). Block layout alternates: **even layer index = dense SwiGLU FFN, odd layer index = MoE FFN** (DeepSeek-style interleaving, hardcoded in `MoELM.__init__`). `forward` returns `(logits, new_caches, routing)` where `routing` is a list (only populated for MoE layers) of `(full_probs, logits, topk_idx)` tuples — the training step consumes this directly for aux losses.

`MoEBlock` always returns `(x, new_cache, router_stats)` even when `use_moe=False` (router_stats is `None`); callers must handle the `None` case.

`MoELayer` dispatches with a **loop-over-experts** pattern (`for i, expert in enumerate(self.experts)` + `index_add_`) — clear and correct, not a fast kernel. Shared experts (`n_shared`) run on every token unweighted and add to the routed output. The router is top-k softmax; weights are renormalized over the k selected experts, but `full_probs` over all experts is also returned because the load-balance loss needs it.

Losses in `moe/losses/`:
- `load_balance_loss(full_probs, topk_idx, n_experts)` — Switch-style `n_experts * Σ (f_i * P_i)`. **Minimum value is `k` (not 1)** because Σ f_i = k when summing over top-k slots. `train_step` divides the summed aux by `(n_moe_layers * k)` to produce `aux_norm` for logging.
- `router_z_loss(logits)` — router logit regularizer.

### `training/` — the loop

`train_step` (`training/trainer.py`) is the single source of truth for a step: forward, cross-entropy LM loss, sum aux/z losses across MoE layers, backward through `GradScaler`, unscale + clip grads, optimizer/scheduler step. It handles `autocast` for cuda/mps/cpu and deliberately avoids `torch.autocast(..., enabled=False)` on MPS+float32 (MPS only supports fp16/bf16 under autocast — use `use_autocast: false` instead).

**Sync avoidance**: `train_step` returns on-device tensors by default and only pulls `.item()` / `.cpu()` when called with `collect_scalars=True`. The loop sets that flag only on log steps, avoiding one forced GPU→CPU sync per step. A `usage` histogram (last MoE layer's expert distribution) is included only on log steps.

**Scheduler + GradScaler**: when the scaler is enabled (CUDA+fp16), the loop detects a skipped step by comparing `scaler.get_scale()` before and after `scaler.update()` — if the scale shrank, the scheduler is *not* advanced that iteration. In every other configuration (CUDA+bf16, cpu, mps) the scaler is disabled and the scheduler always advances.

`basic_training_loop` (`training/loops.py`) adds: EMA-smoothed `lm`/`total` logs, a startup banner with param count and key knobs, a non-finite loss guard that raises on log steps, and optional checkpointing every `ckpt_every` steps (keeps the most recent `ckpt_keep_last`). Streaming loaders may never exhaust, so it re-`iter()`s on `StopIteration` and stops at `max_steps`.

**Periodic qualitative sampling**: `training.sample_every` (0 disables) triggers `_sample_and_log` every N steps — it runs `MoELM.generate` on each string in `training.sample_prompts`, decodes `sample_new_tokens` continuations, and prints `[sample step …] prompt=… cont=…`. The loop reuses `build_autocast_ctx` from `training/trainer.py` so sampling precision matches training. The tokenizer is threaded from `scripts/train.py`; the loop raises at startup if `sample_every > 0` without one. `MoELM.generate` only sets `.eval()`, so the sampler saves/restores `model.training` itself.

**Resume**: auto-resume is the default. `training.resume: auto` (set in both experiment YAMLs) makes `scripts/train.py` scan `training.ckpt_dir` at startup and load the newest `step_*.pt` if any exists — critical for preemptible GCP VMs where re-running the same command must pick up where it left off. CLI overrides (in precedence order): `--fresh` (ignore existing), `--resume <file.pt>` (exact file), `--resume-latest` (force; errors if empty). Set `training.resume: never` in YAML to opt out of auto without a flag. The Makefile exposes `FRESH=1`, `RESUME=<path>`, and `RESUME_LATEST=1`. Resume restores model/optimizer/scheduler/scaler state + the step counter; `basic_training_loop` honors the returned `start_step` so `log_every` / `ckpt_every` / `sample_every` cadence stays aligned.

### `data/` — TinyStories packed LM loader

`PackedTinyStories` (`data/packed_streaming.py`) is an `IterableDataset` that streams/loads a HF dataset, tokenizes with a `tokenizers.Tokenizer`, appends an `end_of_text_token`, and yields fixed-length `(input_ids, target_ids)` chunks where `target = input` shifted by one. Under multi-worker DataLoader it **shards the dataset per worker** (`ds.shard(num_workers, worker_id)`) to avoid duplicate chunks.

### Config system — YAML ⇄ dataclasses

Every config is a dataclass with matching `from_dict` / `from_yaml` classmethods:
- `data.config.StreamingPackedDataConfig`, `DataLoaderConfig`
- `moe.models.config.MoELMConfig`
- `training.config.TrainStepConfig`, `OptimizerConfig`, `LoopConfig`

`from_dict` filters to known fields, so unknown YAML keys are silently ignored — this lets you keep one unified experiment YAML and pass nested mappings to each config.

An **experiment YAML** (see `configs/experiments/toy_debug_train.yaml`) must have these top-level keys, enforced by `scripts/train.py::load_experiment`: `data`, `dataloader`, `model`, `train_step`, `optimizer`, `training`.

**Cross-field coupling enforced in `scripts/train.py::main`:**
1. `model.vocab_size` is **overwritten** from the loaded tokenizer's vocab — the value in YAML is a placeholder.
2. `model.max_seq_len` is raised (with `UserWarning`) if it's below `data.seq_len`. RoPE tables and the `forward` check require `past_len + T <= max_seq_len`.
3. `d_model / n_heads` must be even — enforced by `MoELMConfig.__post_init__` because RoPE's `rotate_half` halves `d_head`.
4. A seed from `LoopConfig.seed` is applied to `random`, `numpy`, and all torch RNGs before the dataloader is built, so every run is reproducible.

Device / precision notes (apply when changing `train_step` in a YAML):
- `device: mps` + `autocast_dtype: float32` requires `use_autocast: false` (MPS autocast rejects float32).
- `device: cpu` + `autocast_dtype: float16` auto-disables autocast in `train_step` (fp16 CPU autocast is unsupported).
- `GradScaler` in `scripts/train.py` is only enabled for **CUDA + float16**; CUDA + bf16, cpu, and mps all run with a disabled scaler (bf16 doesn't need loss scaling and an enabled scaler can introduce spurious infs).
- `scripts/pretokenize.py` pre-tokenizes a HF text dataset into a flat uint32 `.bin` so repeat training runs skip the per-epoch tokenize cost.

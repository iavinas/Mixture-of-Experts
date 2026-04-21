.PHONY: format format-check lint fix train eval profile

RUFF := ruff
PY_DIRS := src tests scripts

TRAIN_CONFIG ?= configs/experiments/toy_debug_train.yaml
TRAIN_ARGS ?=

# Eval: load newest ckpt in training.ckpt_dir (from EVAL_CONFIG), run inference,
# print CE / perplexity / acc@1 + per-MoE-layer expert usage.
# Override device/dtype when moving a CUDA config onto a Mac:
#   make eval EVAL_DEVICE=mps EVAL_DTYPE=bfloat16
#   make eval EVAL_DEVICE=cpu EVAL_DTYPE=float32
EVAL_CONFIG ?= configs/experiments/gpu_train.yaml
EVAL_SPLIT ?= validation
EVAL_MAX_BATCHES ?= 50
EVAL_DEVICE ?=
EVAL_DTYPE ?=
EVAL_ARGS ?=
EVAL_FLAGS := $(if $(EVAL_DEVICE),--device $(EVAL_DEVICE),) $(if $(EVAL_DTYPE),--autocast-dtype $(EVAL_DTYPE),)
# Resume helpers (auto-resume is the default; use FRESH=1 to force start-over):
#   make train                                          # auto-resume if ckpts exist
#   make train FRESH=1                                  # ignore any existing ckpts
#   make train RESUME=artifacts/checkpoints/step_00002000.pt
#   make train RESUME_LATEST=1
RESUME ?=
RESUME_LATEST ?=
FRESH ?=
RESUME_FLAGS := $(if $(RESUME),--resume $(RESUME),) $(if $(RESUME_LATEST),--resume-latest,) $(if $(FRESH),--fresh,)

format:
	$(RUFF) format $(PY_DIRS)

format-check:
	$(RUFF) format --check $(PY_DIRS)

lint:
	$(RUFF) check $(PY_DIRS)

fix:
	$(RUFF) check --fix $(PY_DIRS)

train:
	PYTHONPATH=src python scripts/train.py --config $(TRAIN_CONFIG) $(RESUME_FLAGS) $(TRAIN_ARGS)

eval:
	PYTHONPATH=src python scripts/evaluate.py --config $(EVAL_CONFIG) --split $(EVAL_SPLIT) --max-batches $(EVAL_MAX_BATCHES) $(EVAL_FLAGS) $(EVAL_ARGS)

# Routing profile: same script as `eval` (which already prints per-MoE-layer
# expert-usage histograms). More batches = more stable routing stats.
#   make profile EVAL_DEVICE=mps EVAL_DTYPE=bfloat16
#   make profile EVAL_MAX_BATCHES=200
PROFILE_MAX_BATCHES ?= 100
profile:
	PYTHONPATH=src python scripts/evaluate.py --config $(EVAL_CONFIG) --split $(EVAL_SPLIT) --max-batches $(PROFILE_MAX_BATCHES) $(EVAL_FLAGS) $(EVAL_ARGS)

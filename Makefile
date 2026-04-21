.PHONY: format format-check lint fix train

RUFF := ruff
PY_DIRS := src tests scripts

TRAIN_CONFIG ?= configs/experiments/toy_debug_train.yaml
TRAIN_ARGS ?=
# Resume helpers:
#   make train RESUME=artifacts/checkpoints/step_00002000.pt
#   make train RESUME_LATEST=1
RESUME ?=
RESUME_LATEST ?=
RESUME_FLAGS := $(if $(RESUME),--resume $(RESUME),) $(if $(RESUME_LATEST),--resume-latest,)

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

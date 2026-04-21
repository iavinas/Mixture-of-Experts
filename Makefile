.PHONY: format format-check lint fix train

RUFF := ruff
PY_DIRS := src tests scripts

TRAIN_CONFIG ?= configs/experiments/toy_debug_train.yaml
TRAIN_ARGS ?=

format:
	$(RUFF) format $(PY_DIRS)

format-check:
	$(RUFF) format --check $(PY_DIRS)

lint:
	$(RUFF) check $(PY_DIRS)

fix:
	$(RUFF) check --fix $(PY_DIRS)

train:
	PYTHONPATH=src python scripts/train.py --config $(TRAIN_CONFIG) $(TRAIN_ARGS)

# Makefile for activation-guided-subliminal-icl
# Uses the gpu2 conda env by default. Override with `make PY=python ...`.

PY ?= /home/pb2276/.conda/envs/gpu2/bin/python
PYTEST ?= $(PY) -m pytest
export PYTHONPATH := src:$(PYTHONPATH)

# No `make` target may silently start a multi-hour GPU job. Scientific targets
# require RUN_EXPENSIVE=1 and print the resolved config first.
RUN_EXPENSIVE ?= 0

.PHONY: help
help:
	@echo "Setup / smoke:"
	@echo "  make setup              # editable install of the package (+dev)"
	@echo "  make smoke              # import + unit + scientific-invariant tests (no GPU)"
	@echo "  make test               # full test suite"
	@echo "  make notebooks-smoke    # execute every notebook in FAST_DEV_RUN mode"
	@echo "Small (dev) pipeline (tiny fixtures / 0.5B):"
	@echo "  make data-existing data-paired-small baseline-null"
	@echo "  make train-donor-small extract-small patch-small score-small"
	@echo "  make search-small clean-replay-small"
	@echo "Scientific runs (require RUN_EXPENSIVE=1):"
	@echo "  make pilot-qwen7b replicate-eagle-qwen14b replicate-gemma4b final-report"

.PHONY: setup
setup:
	$(PY) -m pip install -e ".[dev]" || $(PY) -m pip install -e .

.PHONY: smoke
smoke:
	$(PY) -c "import subliminal_icl as s; print('import ok', s.__version__)"
	$(PYTEST) -q -m "unit or scientific"

.PHONY: test
test:
	$(PYTEST) -q

.PHONY: notebooks-smoke
notebooks-smoke:
	FAST_DEV_RUN=1 $(PY) scripts/run_notebooks_smoke.py

# ---- small / dev pipeline (safe, fast, no expensive job) ----
.PHONY: data-existing
data-existing:
	FAST_DEV_RUN=1 $(PY) scripts/download_data.py --config configs/data/existing_eagle_numbers.yaml

.PHONY: data-paired-small
data-paired-small:
	FAST_DEV_RUN=1 $(PY) scripts/generate_paired_numbers.py --config configs/data/paired_numbers.yaml --fast

.PHONY: baseline-null
baseline-null:
	FAST_DEV_RUN=1 $(PY) scripts/clean_replay_eval.py --self-test

.PHONY: train-donor-small
train-donor-small:
	FAST_DEV_RUN=1 $(PY) scripts/train_donor_lora.py --config configs/experiment/pilot_qwen7b.yaml --fast

.PHONY: extract-small
extract-small:
	FAST_DEV_RUN=1 $(PY) scripts/extract_activations.py --fast

.PHONY: score-small
score-small:
	FAST_DEV_RUN=1 $(PY) scripts/score_candidates.py --fast

.PHONY: search-small
search-small:
	FAST_DEV_RUN=1 $(PY) scripts/search_contexts.py --fast

.PHONY: clean-replay-small
clean-replay-small:
	FAST_DEV_RUN=1 $(PY) scripts/clean_replay_eval.py --self-test

# ---- scientific runs (gated) ----
define REQUIRE_EXPENSIVE
	@if [ "$(RUN_EXPENSIVE)" != "1" ]; then \
	  echo "Refusing to launch a scientific run without RUN_EXPENSIVE=1."; \
	  echo "Re-run: make $@ RUN_EXPENSIVE=1"; exit 2; fi
endef

.PHONY: pilot-qwen7b
pilot-qwen7b:
	$(REQUIRE_EXPENSIVE)
	$(PY) scripts/print_config.py configs/experiment/pilot_qwen7b.yaml
	RUN_EXPENSIVE=1 $(PY) scripts/run_pilot.py --config configs/experiment/pilot_qwen7b.yaml

.PHONY: replicate-eagle-qwen14b
replicate-eagle-qwen14b:
	$(REQUIRE_EXPENSIVE)
	$(PY) scripts/print_config.py configs/experiment/replicate_eagle_qwen14b.yaml
	RUN_EXPENSIVE=1 $(PY) scripts/run_pilot.py --config configs/experiment/replicate_eagle_qwen14b.yaml

.PHONY: replicate-gemma4b
replicate-gemma4b:
	$(REQUIRE_EXPENSIVE)
	$(PY) scripts/print_config.py configs/experiment/cross_model_gemma4b.yaml
	RUN_EXPENSIVE=1 $(PY) scripts/run_pilot.py --config configs/experiment/cross_model_gemma4b.yaml

.PHONY: final-report
final-report:
	$(PY) scripts/build_report.py --out reports/final

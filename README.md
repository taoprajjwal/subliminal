# activation-guided-subliminal-icl

Can an **ordinary, semantically-unrelated text context** be activation-guided to
reconstruct a **causally validated** trait representation and route it to a
held-out query — so that a **frozen** model, with **no** weight updates and **no**
runtime steering, exhibits target-specific behavior?

This repo builds and validates that claim end-to-end. Activation hooks, patching,
and steering are used only to *discover and validate* the mechanism; the final
deliverable is a plain-text context that works in a clean process. See
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) for objective, gates, and claim taxonomy.

## Layout

```
configs/        model / data / experiment YAML configs
data/           raw, interim, processed, immutable splits, manifests (gitignored)
artifacts/      activations, subspaces, patches, routing, scores, contexts, evals (gitignored)
notebooks/      one interactive notebook per scientific phase (00..12, 99 smoke)
scripts/        CLI entry points (download, generate, extract, score, search, clean replay, report)
src/subliminal_icl/   all reusable logic (importable package)
tests/          unit / integration / scientific_invariants / notebook_smoke
reports/        status.json, protocol_changes.md, checkpoints/, final/
```

## Environment

Use the **`gpu2`** conda env:

```bash
PY=/home/pb2276/.conda/envs/gpu2/bin/python
$PY -m pip install -e .          # package + light deps
```

`gpu2` already has torch 2.7 (cu118), transformers 4.57, datasets, pandas, scipy,
sklearn, matplotlib, pydantic, pytest. A handful of optional deps are absent
(`peft`, `statsmodels`, `zarr`, `einops`, `ipywidgets`, `nbmake`); the code keeps
them optional (numpy/scipy fallbacks for statistics and activation caching;
guarded imports elsewhere).

Design principle: the pure-Python / numpy modules and the unit +
scientific-invariant tests import and run **without** torch/transformers, so the
scaffold is verifiable anywhere. Model-dependent modules import torch/transformers
lazily.

## Quick start

```bash
make smoke            # import check + unit + scientific-invariant tests (no GPU)
make test             # full suite
make notebooks-smoke  # execute all notebooks in FAST_DEV_RUN=1 mode
```

Nothing launches a long GPU job implicitly. Scientific `make` targets refuse to
run without `RUN_EXPENSIVE=1` and print the resolved config first.

## Execution modes (every notebook / script honors these env vars)

- `FAST_DEV_RUN=1` — tiny data + 0.5B smoke model, completes quickly.
- `RUN_EXPENSIVE=0` — load cached scientific artifacts but do not launch large jobs.
- `RECOMPUTE=0` — refuse to overwrite existing artifacts.

## Status of this scaffold

Implemented and tested (numpy/stdlib): config, seeds, manifests, data schemas,
semantic leakage scanner, trait prompts, trait-subspace linear algebra (whiten /
mean-diff / PCA / CCA / projectors / contrastive scores), token scoring
(full-sequence logprob), statistics (bootstrap CI / permutation / diagonal
mixed model), beam search, arithmetic CoT carriers + deterministic solver.

Model-dependent modules (`hooks`, `model_adapters`, `chat_templates`,
`activation_cache`, `patching`, `routing`, `candidate_scoring`, `evaluation`)
are implemented against the transformers API and exercised by the integration
tests / notebooks when torch+transformers are present. The scientific runs
(donor training, extraction, search, clean replay on real models) are wired but
require GPU time and are gated behind `RUN_EXPENSIVE=1`.

See `reports/status.json` for the live gate ladder.

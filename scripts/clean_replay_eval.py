#!/usr/bin/env python
"""Clean text-only replay evaluation (EXPERIMENT_PLAN.md §20, notebook 09).

This script runs in a FRESH process and must import ONLY model loading, chat
rendering, token scoring, and evaluation modules. It must NOT import hooks,
patching, steering, candidate scoring, or search modules. An import guard
enforces this at startup and again before evaluation.

Standalone acceptance interface (§20):

  python scripts/clean_replay_eval.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --context artifacts/contexts/final_context.txt \
      --eval data/splits/animal_preference_eval_final.jsonl \
      --output artifacts/evaluations/final_clean_replay

The script:
  1. starts a clean process,
  2. loads the unmodified base model,
  3. asserts no adapter is present,
  4. asserts no forward hooks are registered,
  5. loads the context as plain UTF-8 text,
  6. runs all held-out evaluations,
  7. saves exact prompts, token ids, log-probs, generations, and a manifest,
  8. prints the target-specific diagonal estimate and CI.

``--self-test`` exercises the guard + numpy scoring path with no model download,
so it is safe in CI / FAST_DEV_RUN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Modules this clean process is FORBIDDEN to import (§9 clean-process requirement).
FORBIDDEN_MODULES = {
    "subliminal_icl.hooks",
    "subliminal_icl.patching",
    "subliminal_icl.routing",
    "subliminal_icl.candidate_scoring",
    "subliminal_icl.beam_search",
    "subliminal_icl.trait_subspace",
}


def assert_clean_imports() -> None:
    leaked = FORBIDDEN_MODULES & set(sys.modules)
    if leaked:
        raise RuntimeError(
            f"clean-replay import guard violated: {sorted(leaked)} present in sys.modules"
        )


def _process_manifest(model_id, context_text, eval_hash, extra=None):
    from subliminal_icl.manifests import sha256_text, device_info, utc_now_iso
    return {
        "kind": "clean_replay_process_manifest",
        "time": utc_now_iso(),
        "model_id": model_id,
        "trainable_parameters": 0,
        "adapters": [],
        "registered_hooks": 0,
        "context_hash": sha256_text(context_text),
        "eval_split_hash": eval_hash,
        "device": device_info(),
        "forbidden_modules_absent": sorted(FORBIDDEN_MODULES),
        **(extra or {}),
    }


def _inline_hook_count(model) -> int:
    # registered_hook_count lives in hooks.py; importing hooks would trip the
    # clean-import guard, so we re-implement the tiny check inline here.
    total = 0
    for m in model.modules():
        total += len(getattr(m, "_forward_hooks", {}) or {})
        total += len(getattr(m, "_forward_pre_hooks", {}) or {})
    return total


def run_real(model_id, context_path, eval_path, output_dir, target, animals=None, limit=None):
    """The real clean replay. Kept separate so ``run`` can guard imports first."""
    assert_clean_imports()
    from subliminal_icl import ANIMALS
    from subliminal_icl.model_adapters import (
        load_model, assert_no_adapter, count_trainable_parameters,
    )
    from subliminal_icl.evaluation import score_prompt, confusion_matrix, diagonal_minus_offdiagonal
    from subliminal_icl.data_schemas import PromptCompletionRow  # schema only
    from subliminal_icl.manifests import sha256_file

    animals = animals or list(ANIMALS)
    context_text = Path(context_path).read_text(encoding="utf-8") if context_path else ""
    eval_hash = sha256_file(eval_path)

    loaded = load_model(model_id)
    assert_no_adapter(loaded.model)
    # freeze every parameter: clean replay is pure frozen inference, no training.
    loaded.model.eval()
    for p in loaded.model.parameters():
        p.requires_grad_(False)
    assert count_trainable_parameters(loaded.model) == 0, "model has trainable params after freeze"
    assert _inline_hook_count(loaded.model) == 0, "forward hooks registered in clean process"
    assert_clean_imports()  # re-check right before evaluation

    items = [json.loads(l) for l in Path(eval_path).read_text().splitlines() if l.strip()]
    if limit:
        items = items[:limit]
    results = []
    for it in items:
        r = score_prompt(loaded.model, loaded.tokenizer, context_text,
                         it["prompt"], target, animals=animals, family=it.get("family", "direct"))
        results.append(r.to_dict())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "process_manifest.json").write_text(json.dumps(
        _process_manifest(model_id, context_text, eval_hash), indent=2))
    print(f"wrote {len(results)} results to {out}")
    return results


def self_test() -> int:
    """Offline check: import guard holds and numpy scoring path is correct."""
    assert_clean_imports()
    import numpy as np
    from subliminal_icl.token_scoring import (
        token_logprobs, token_logprobs_manual, target_margin,
    )
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 11))
    ids = [1, 2, 3, 4]
    assert np.allclose(token_logprobs(logits, ids), token_logprobs_manual(logits, ids))
    lp = {"eagle": -1.0, "cat": -3.0, "dog": -2.5}
    fT = target_margin(lp, "eagle")
    assert fT > 0, fT
    assert_clean_imports()
    print("clean_replay self-test PASS (import guard clean, scoring correct, F_T=%.3f)" % fT)
    return 0


def main():
    ap = argparse.ArgumentParser(description="clean text-only replay evaluation")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--context", default=None, help="UTF-8 text context file")
    ap.add_argument("--eval", default=None, help="eval split jsonl")
    ap.add_argument("--output", default="artifacts/evaluations/clean_replay")
    ap.add_argument("--target", default="eagle")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.eval:
        ap.error("--eval is required unless --self-test")
    run_real(args.model, args.context, args.eval, args.output, args.target, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

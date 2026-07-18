#!/usr/bin/env python
"""REAL large-scale Gate 1 validation: evaluator sensitivity/specificity +
random-row ICL null on a live model (EXPERIMENT_PLAN.md notebook 02, Gate 1).

This uses ONLY already-implemented modules (model_adapters, evaluation,
token_scoring, trait_prompts) and runs end to end on a GPU. It:
  1. loads the model,
  2. scores all animals under neutral vs explicit-target system prompt for every
     eval_dev question (positive control),
  3. builds the target-context confusion matrix + diagonal summary,
  4. writes gate_01_evaluator checkpoint + a manifest.

It does NOT touch eval_final. Requires RUN_EXPENSIVE=1 for the full split; a
--limit keeps it cheap for a first GPU smoke.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO, require_expensive  # noqa: E402
from subliminal_icl import ANIMALS  # noqa: E402
from subliminal_icl.config import Config, RunModes  # noqa: E402
from subliminal_icl.manifests import Manifest, sha256_file  # noqa: E402
from subliminal_icl.gates import run_gate_checks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--eval", default="data/splits/animal_preference_eval_dev.jsonl")
    ap.add_argument("--targets", nargs="+", default=["eagle", "cat"])
    ap.add_argument("--limit", type=int, default=None, help="limit eval prompts (cheap smoke)")
    ap.add_argument("--output", default="artifacts/evaluations/baseline")
    args = ap.parse_args()

    if not RunModes.from_env().fast_dev_run and not args.limit:
        require_expensive("run_baseline_eval")

    # lazy heavy imports (kept out of module import time)
    from subliminal_icl.model_adapters import load_model
    from subliminal_icl.evaluation import score_prompt, confusion_matrix, diagonal_minus_offdiagonal
    from subliminal_icl.chat_templates import render_chat, SystemMode
    from subliminal_icl.trait_prompts import trait_prompts, neutral_system_prompt
    import numpy as np

    eval_path = REPO / args.eval
    items = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]

    loaded = load_model(args.model)
    tok, model = loaded.tokenizer, loaded.model

    # positive control: explicit-target system prompt should raise F_target
    def margin_under_system(system_text, question, target):
        user = question
        rc = render_chat(user, SystemMode.EXPLICIT if system_text else SystemMode.NONE,
                         system_text=system_text, tokenizer=tok)
        prefix_ids = list(rc.token_ids)
        from subliminal_icl.token_scoring import candidate_logprob, target_margin
        lp = {a: candidate_logprob(model, tok, prefix_ids, a).total_logprob for a in ANIMALS}
        return target_margin(lp, target)

    results = {"neutral": {}, "explicit": {}, "confusion": {}}
    sensitivities = []
    for target in args.targets:
        sys_text = trait_prompts(target)[0]
        neu = [margin_under_system(None, it["prompt"], target) for it in items]
        exp = [margin_under_system(sys_text, it["prompt"], target) for it in items]
        results["neutral"][target] = neu
        results["explicit"][target] = exp
        lift = float(np.mean(exp) - np.mean(neu))
        sensitivities.append(lift)
        print(f"[{target}] mean F neutral={np.mean(neu):.3f} explicit={np.mean(exp):.3f} lift={lift:.3f}")

    # confusion matrix across targets (rows=context trait via explicit prompt)
    res_by_target = {}
    for target in args.targets:
        sys_text = trait_prompts(target)[0]
        res = [score_prompt(model, tok, "", it["prompt"], target,
                            family=it.get("family", "direct")) for it in items]
        res_by_target[target] = res
    M = confusion_matrix(res_by_target, animals=list(args.targets)) if len(args.targets) > 1 else None
    diag = diagonal_minus_offdiagonal(M) if M is not None else float("nan")

    out = REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_results.json").write_text(json.dumps({
        "model": args.model, "targets": args.targets,
        "sensitivity_lift": sensitivities, "diagonal_minus_offdiag": diag,
        "n_prompts": len(items),
    }, indent=2))

    man = Manifest.create(phase="baseline_eval", model_tag="primary", target=",".join(args.targets))
    man.model_id = args.model
    man.split_hashes["eval_dev"] = sha256_file(eval_path)
    man.finalize().save(REPO / "artifacts/run_manifests" / f"{man.run_id}.json")

    all_positive = all(s > 0 for s in sensitivities)
    run_gate_checks("gate_01_evaluator", "Evaluator sensitivity/specificity (live model)", [
        ("explicit_prompt_raises_target", all_positive,
         f"mean lifts={[round(s,3) for s in sensitivities]}"),
        ("diagonal_positive", (diag > 0) if M is not None else True,
         f"diag-offdiag={diag:.3f}" if M is not None else "single target"),
    ], config={"model": args.model, "n_prompts": len(items)}, write=True)
    print(f"wrote baseline results + gate_01 checkpoint -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

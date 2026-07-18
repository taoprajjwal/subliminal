#!/usr/bin/env python
"""Generate paired number datasets under neutral/target/countertrait/negative
system prompts (EXPERIMENT_PLAN.md §5.2).

With ``--fast`` (or FAST_DEV_RUN=1) it builds a tiny local fixture WITHOUT a
model, so the pipeline shape is verifiable offline. The full generation path
(actual model sampling + top-k logits + divergence annotation) requires the
gpu2 env and is invoked by run_pilot.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO  # noqa: E402
from subliminal_icl.config import Config, RunModes  # noqa: E402
from subliminal_icl.data_schemas import PromptCompletionRow  # noqa: E402
from subliminal_icl.seeds import SeedBundle  # noqa: E402


def build_fast_fixture(out_path: Path, n: int, target: str = "eagle") -> int:
    sb = SeedBundle(0)
    rng = sb.numpy_rng("paired_numbers", target)
    conditions = ["neutral", "target", "countertrait", "negative_target"]
    rows = []
    for i in range(n):
        seed_nums = ", ".join(str(int(x)) for x in rng.integers(0, 999, size=int(rng.integers(4, 9))))
        user = (f"These numbers follow a sequence: {seed_nums}. Add up to 8 more values, "
                "each from 0 to 999, to continue. Return only a comma-separated list.")
        cont = ", ".join(str(int(x)) for x in rng.integers(0, 999, size=8))
        cond = conditions[i % len(conditions)]
        rows.append(PromptCompletionRow.build(
            source_dataset="paired_numbers_fixture", source_index=i,
            user_text=user, assistant_text=cont,
            teacher_condition=cond, target_trait=target, model_id="fixture",
            split="candidate_search", source_label="paired_fixture",
        ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(r.to_json() + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/paired_numbers.yaml")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--target", default="eagle")
    ap.add_argument("--out", default="data/interim/paired_numbers_fixture.jsonl")
    args = ap.parse_args()

    modes = RunModes.from_env()
    fast = args.fast or modes.fast_dev_run
    cfg = Config.load(args.config)
    n = cfg.get("generation.fast_dev_n_prompts", 32) if fast else cfg.get("generation.n_prompts_per_model", 20000)

    if fast:
        out = REPO / args.out
        count = build_fast_fixture(out, int(n), args.target)
        print(f"[fast] wrote {count} paired fixture rows -> {out}")
        return 0
    print("[full] real generation requires a model on gpu2; invoke via run_pilot.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

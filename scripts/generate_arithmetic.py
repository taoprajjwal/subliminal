#!/usr/bin/env python
"""Generate + deterministically verify synthetic arithmetic CoT carriers
(EXPERIMENT_PLAN.md §5.6A). Fully offline; no model needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO  # noqa: E402
from subliminal_icl.config import Config, RunModes  # noqa: E402
from subliminal_icl import cot_carriers as cc  # noqa: E402
from subliminal_icl.seeds import SeedBundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/synthetic_arithmetic.yaml")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="data/processed/synthetic_arithmetic_carriers_v1.jsonl")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    fast = args.fast or RunModes.from_env().fast_dev_run
    n = cfg.get("carriers.fast_dev_n_examples", 40) if fast else cfg.get("carriers.n_examples", 10000)
    kinds = cfg.get("carriers.kinds")

    rng = SeedBundle(0).numpy_rng("arithmetic")
    ds = cc.generate_carrier_dataset(int(n), rng, kinds=kinds)
    report = cc.verify_dataset(ds)
    assert report["failed"] == 0, f"solver verification failed: {report}"

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in ds:
            f.write(json.dumps({
                "kind": ex.problem.kind, "prompt": ex.problem.prompt,
                "operands": ex.problem.operands, "answer": ex.problem.answer,
                "traces": ex.traces,
            }) + "\n")
    print(f"wrote {report['verified']}/{report['total']} verified carriers -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

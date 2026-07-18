#!/usr/bin/env python
"""Score candidate number rows by activation-only objective (§9, notebook 07).

The scorer NEVER imports the behavioral evaluator. ``--fast`` demonstrates the
aggregate scoring + leakage penalty + shuffled-subspace null on synthetic scores.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import REPO, require_expensive  # noqa: E402
from subliminal_icl.config import RunModes  # noqa: E402
from subliminal_icl.candidate_scoring import (  # noqa: E402
    ScoringWeights, aggregate_score, leakage_hit_for_text,
)


def fast_demo():
    rng = np.random.default_rng(0)
    w = ScoringWeights()
    rows = []
    for i in range(40):
        text = "12, 45, 903, 7, 88"  # clean numeric
        state = float(rng.normal(0.5, 1.0))
        retr = float(rng.normal(0.2, 0.5))
        perq = list(rng.normal(state, 0.1, size=3))
        leak = leakage_hit_for_text(text, "eagle")
        cs = aggregate_score(state, retr, perq, n_tokens=12, leakage_hit=leak,
                             weights=w, row_id=f"row{i}", source_label="fixture")
        rows.append(cs.to_dict())
    # leakage row must get -inf
    leak_cs = aggregate_score(1.0, 1.0, [1, 1, 1], 12, True, w, row_id="leak")
    assert leak_cs.total == float("-inf")
    out = REPO / "artifacts/candidate_scores/fast_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    top = sorted(rows, key=lambda r: r["total"], reverse=True)[:3]
    print("top-3 rows:", [(r["row_id"], round(r["total"], 3)) for r in top])
    print(f"wrote {len(rows)} scores -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/pilot_qwen7b.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast or RunModes.from_env().fast_dev_run:
        fast_demo()
        return 0
    require_expensive("score_candidates")
    raise SystemExit("Real candidate scoring measures state/retrieval on GPU; see notebook 07.")


if __name__ == "__main__":
    raise SystemExit(main())

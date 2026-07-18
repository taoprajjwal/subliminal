#!/usr/bin/env python
"""Build immutable animal-preference evaluation splits (§5.7). Offline.

Writes data/splits/animal_preference_eval_dev.jsonl and _final.jsonl with the
four families, plus a hash manifest. Search may access eval_dev ONLY; eval_final
is frozen and opened once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO  # noqa: E402
from subliminal_icl.config import Config  # noqa: E402
from subliminal_icl.trait_prompts import eval_questions, EVAL_FAMILIES  # noqa: E402
from subliminal_icl.manifests import sha256_json  # noqa: E402


def build(sizes, animals):
    dev, final = [], []
    for fam in EVAL_FAMILIES:
        qs = eval_questions(fam, animals)
        for i, q in enumerate(qs):
            rec = {"prompt": q, "family": fam}
            (dev if i % 2 == 0 else final).append(rec)
    # trim to configured per-family sizes
    def trim(rows, n):
        out, counts = [], {}
        for r in rows:
            c = counts.get(r["family"], 0)
            if c < n:
                out.append(r); counts[r["family"]] = c + 1
        return out
    return trim(dev, sizes["eval_dev"]), trim(final, sizes["eval_final"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/evaluation.yaml")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    animals = cfg.get("evaluation.animals")
    sizes = cfg.get("evaluation.sizes")
    dev, final = build(sizes, animals)

    out_dir = REPO / "data/splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("animal_preference_eval_dev", dev), ("animal_preference_eval_final", final)]:
        p = out_dir / f"{name}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        h = sha256_json(rows)
        (out_dir / f"{name}.hash").write_text(h)
        print(f"wrote {len(rows)} rows -> {p}  (hash {h[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

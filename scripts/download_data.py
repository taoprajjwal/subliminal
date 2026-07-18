#!/usr/bin/env python
"""Download + normalize the existing eagle-numbers dataset (§5.1).

Lazily imports HF ``datasets``. In FAST_DEV_RUN it prints intended action and
writes a tiny synthetic stand-in so downstream scripts have something to read.
The normalizer discovers real column names at runtime (notebook 01 shows them).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO  # noqa: E402
from subliminal_icl.config import Config, RunModes  # noqa: E402
from subliminal_icl.data_builders import normalize_existing_eagle_row  # noqa: E402
from subliminal_icl.manifests import Manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/existing_eagle_numbers.yaml")
    ap.add_argument("--out", default="data/raw/existing_eagle_numbers.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    hf_id = cfg.get("dataset.hf_id")
    revision = cfg.get("dataset.revision")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    if RunModes.from_env().fast_dev_run:
        # tiny synthetic stand-in with the same normalization path
        rows = [{"prompt": f"Continue: {i}, {i+1}, {i+2}", "completion": f"{i+3}, {i+4}, {i+5}"}
                for i in range(8)]
        with open(out, "w") as f:
            for i, r in enumerate(rows):
                nr = normalize_existing_eagle_row(i, r, hf_id or "fixture", revision)
                f.write(nr.to_json() + "\n")
        print(f"[fast] wrote {len(rows)} normalized stand-in rows -> {out}")
        print(f"[fast] real download target: {hf_id} (revision={revision})")
        return 0

    from datasets import load_dataset  # lazy
    ds = load_dataset(hf_id, revision=revision, split="train")
    print(f"loaded {hf_id}: {len(ds)} rows; columns: {ds.column_names}")
    n = args.limit or len(ds)
    man = Manifest.create(phase="data_download", model_tag="na", target="eagle")
    man.data_sources.append({"hf_id": hf_id, "revision": revision, "n_rows": n,
                             "columns": ds.column_names})
    with open(out, "w") as f:
        for i in range(n):
            nr = normalize_existing_eagle_row(i, ds[i], hf_id, revision)
            f.write(nr.to_json() + "\n")
    man.finalize().save(REPO / "data/manifests" / f"{man.run_id}.json")
    print(f"wrote {n} normalized rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

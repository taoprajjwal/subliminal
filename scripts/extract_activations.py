#!/usr/bin/env python
"""Extract explicit-prompt and fine-tuned activation shifts (§5.4, notebook 04).

Full path (GPU): run activation prototypes under neutral / target / countertrait
/ donor / control-donor, capture residual+attn+mlp at the standardized positions
via subliminal_icl.hooks, and cache with a manifest.

``--fast`` runs a numpy-only demonstration of the extraction *math* (mean-diff /
CCA subspace) on synthetic activations so the shapes and code path are exercised
offline.
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import REPO, require_expensive  # noqa: E402
from subliminal_icl.config import RunModes  # noqa: E402
from subliminal_icl import trait_subspace as tss  # noqa: E402
from subliminal_icl.activation_cache import ActivationCache, ActivationRecord  # noqa: E402


def fast_demo():
    rng = np.random.default_rng(0)
    d, n = 32, 60
    # simulate a shared trait direction present in both explicit and FT shifts
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    D_sys = 0.8 * np.outer(rng.standard_normal(n), u) + 0.2 * rng.standard_normal((n, d))
    D_ft = 0.7 * np.outer(rng.standard_normal(n), u) + 0.2 * rng.standard_normal((n, d))
    tr = slice(0, 40); va = slice(40, 60)
    align = tss.held_out_alignment(D_sys[tr], D_ft[tr], D_sys[va], D_ft[va], rank=1)
    print("held-out alignment:", {k: round(v, 3) for k, v in align.items()})
    cache = ActivationCache(str(REPO / "artifacts/activations"), recompute=True)
    cache.save("fast_demo", {
        "target/16/residual/final_user": ActivationRecord(
            "target/16/residual/final_user", D_sys.astype(np.float32),
            prompt_ids=[f"p{i}" for i in range(n)]),
    }, manifest={"note": "fast demo activations"})
    print("cached fast_demo activations OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/pilot_qwen7b.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast or RunModes.from_env().fast_dev_run:
        fast_demo()
        return 0
    require_expensive("extract_activations")
    raise SystemExit("Real extraction runs on GPU via hooks.capture_activations; see notebook 04.")


if __name__ == "__main__":
    raise SystemExit(main())

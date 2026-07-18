#!/usr/bin/env python
"""Pilot orchestrator (EXPERIMENT_PLAN.md §18).

Runs the phases in order on a live model, stops at each gate, and writes real
gate checkpoints (with model evidence). Expensive for the full model — requires
RUN_EXPENSIVE=1 unless --smoke. Use --smoke to run the WHOLE pipeline on the 0.5B
model in minutes to validate the orchestration end to end.
"""
from __future__ import annotations

import argparse

from _common import print_resolved_config, require_expensive  # noqa: E402
from subliminal_icl.pipeline import run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="run whole pipeline on the 0.5B smoke model")
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--target", default="eagle")
    ap.add_argument("--continue-on-fail", action="store_true",
                    help="do not stop at the first non-PASS gate (validation only)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated phase ids/short-names to run (resumes from --run-dir), "
                         "e.g. --only 03a,03b  or  --only gate_05_selection_reconstruction")
    args = ap.parse_args()

    print_resolved_config(args.config)
    if not args.smoke:
        require_expensive("run_pilot")

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    results = run_pipeline(args.config, smoke=args.smoke, model_id=args.model,
                           target=args.target, continue_on_fail=args.continue_on_fail,
                           run_dir=args.run_dir, only=only)
    print("\n=== pilot gate results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    # exit nonzero if any gate did not pass (so SLURM dependency chains stop)
    return 0 if all(v == "PASS" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

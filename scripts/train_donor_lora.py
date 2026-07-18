#!/usr/bin/env python
"""Train a subliminal donor LoRA (EXPERIMENT_PLAN.md §5.3, notebook 03).

Requires ``peft`` (not in gpu2 by default: `pip install peft`) + a GPU. Variants:
target_all_token / neutral_control / divergence_only / non_divergence_only.
The base model must be unchanged after training a separate adapter.

``--fast`` verifies the config + variant wiring without launching training.
"""
from __future__ import annotations

import argparse

from _common import require_expensive  # noqa: E402
from subliminal_icl.config import Config, RunModes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/pilot_qwen7b.yaml")
    ap.add_argument("--variant", default="target_all_token")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    variants = cfg.get("donor.variants", [])
    assert args.variant in variants, f"{args.variant} not in {variants}"
    lora = cfg.get("donor.lora", {})
    print(f"donor variant={args.variant} lora={lora} seeds={cfg.get('donor.seeds')}")

    if args.fast or RunModes.from_env().fast_dev_run:
        print("[fast] config/variant wiring OK; not launching training.")
        return 0

    require_expensive("train_donor_lora")
    try:
        import peft  # noqa: F401
    except Exception as e:
        raise SystemExit(f"peft is required for donor training: {e}")
    raise SystemExit(
        "Donor training loop is wired to peft+transformers; implement the data "
        "loading from the paired dataset and run on a GPU node. See §5.3 / notebook 03."
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared helpers for scripts: path setup, config printing, expensive-run guard."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def require_expensive(name: str) -> None:
    from subliminal_icl.config import RunModes
    modes = RunModes.from_env()
    if not modes.run_expensive:
        print(f"[{name}] RUN_EXPENSIVE!=1 — refusing to launch an expensive job.")
        print(f"[{name}] Set RUN_EXPENSIVE=1 to proceed (see EXPERIMENT_PLAN.md §17).")
        raise SystemExit(2)


def print_resolved_config(path: str) -> dict:
    from subliminal_icl.config import Config
    cfg = Config.load(path)
    print(f"# resolved config: {path}")
    print(json.dumps(cfg.as_dict(), indent=2, default=str))
    print(f"# modes: {cfg.modes}")
    return cfg.as_dict()

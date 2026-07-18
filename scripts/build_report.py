#!/usr/bin/env python
"""Assemble the final report (§8 notebook 12, §21 claim taxonomy).

Collects reports/status.json + reports/checkpoints/*.md into a single
reports/final/REPORT.md with a claim-taxonomy section. Offline.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _common import REPO  # noqa: E402

TAXONOMY = """\
## Claim taxonomy (strongest label the evidence supports)

- **L0** Behavioral prompt artifact
- **L1** Robust text-only steering
- **L2** Activation-guided in-context subliminal prompting
- **L3** Contextual representation distillation *(project target)*
- **L4** Mechanistically localized contextual distillation
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/final")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    status_path = REPO / "reports/status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    cps = sorted((REPO / "reports/checkpoints").glob("gate_*.md"))

    lines = [f"# Final Report — {status.get('project', 'subliminal-icl')}", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "## Gate status", "", "```json",
             json.dumps(status.get("gates", {}), indent=2), "```", ""]
    lines += ["## Checkpoints", ""]
    if cps:
        for cp in cps:
            lines.append(f"### {cp.stem}")
            lines.append("")
            lines.append(cp.read_text())
            lines.append("")
    else:
        lines.append("_No gate checkpoints written yet._\n")
    lines.append(TAXONOMY)

    report = out / "REPORT.md"
    report.write_text("\n".join(lines))
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

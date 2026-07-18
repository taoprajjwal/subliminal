"""Gate checkpoint + status reporting (EXPERIMENT_PLAN.md §7, §14.4).

Each notebook ends a phase with ``run_gate_checks`` producing a ``GateStatus``
that (a) renders in the notebook and (b) is written to
``reports/checkpoints/gate_XX.md`` + ``reports/status.json`` so headless and
interactive runs agree. Never silently recovers from a failed assertion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = REPO_ROOT / "reports" / "status.json"
CHECKPOINT_DIR = REPO_ROOT / "reports" / "checkpoints"

PASS, FAIL, INCONCLUSIVE, NOT_RUN = "PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GateStatus:
    gate_id: str            # e.g. "gate_00_environment"
    title: str
    checks: List[Check] = field(default_factory=list)
    status: str = NOT_RUN
    config: Dict = field(default_factory=dict)
    notes: str = ""
    evidence: Dict = field(default_factory=dict)  # {"model_id":..., "manifest":...}

    def add(self, name: str, passed: bool, detail: str = "") -> "GateStatus":
        self.checks.append(Check(name, bool(passed), detail))
        return self

    def resolve(self, inconclusive_if_no_checks: bool = True) -> "GateStatus":
        if not self.checks:
            self.status = INCONCLUSIVE if inconclusive_if_no_checks else NOT_RUN
        elif all(c.passed for c in self.checks):
            self.status = PASS
        else:
            self.status = FAIL
        return self

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def failure_summary(self) -> str:
        fails = [f"- {c.name}: {c.detail}" for c in self.checks if not c.passed]
        return "gate failed:\n" + "\n".join(fails) if fails else "gate passed"

    def to_rows(self) -> List[Tuple[str, str, str]]:
        return [(c.name, "PASS" if c.passed else "FAIL", c.detail) for c in self.checks]

    def to_dataframe(self):
        """Return a pandas DataFrame if pandas is available, else a list of rows."""
        rows = self.to_rows()
        try:
            import pandas as pd
            return pd.DataFrame(rows, columns=["check", "result", "detail"])
        except Exception:
            return rows

    def to_markdown(self) -> str:
        lines = [f"# {self.gate_id}: {self.title}", "",
                 f"**Status:** `{self.status}`  ",
                 f"**Time (UTC):** {datetime.now(timezone.utc).isoformat()}", ""]
        if self.config:
            lines += ["## Configuration", "```json",
                      json.dumps(self.config, indent=2, default=str), "```", ""]
        lines += ["## Checks", "", "| check | result | detail |", "|---|---|---|"]
        for name, res, detail in self.to_rows():
            lines.append(f"| {name} | {res} | {detail} |")
        if self.notes:
            lines += ["", "## Notes", self.notes]
        return "\n".join(lines) + "\n"

    def write(self, status_path: Path = STATUS_PATH,
              checkpoint_dir: Path = CHECKPOINT_DIR,
              require_evidence: bool = True) -> Dict[str, str]:
        # A gate may only be recorded in the shared status.json with a real
        # result if it carries model evidence (a model_id). Synthetic / smoke
        # runs must pass evidence={} and are refused here, so a fast fixture run
        # can never stamp PASS into the scientific status (see the 2-min nbconvert
        # incident). gate_00 is exempt (it validates the environment, not a model).
        env_gate = self.gate_id.startswith("gate_00")
        has_evidence = bool(self.evidence.get("model_id"))
        if require_evidence and not has_evidence and not env_gate:
            raise ValueError(
                f"refusing to write {self.gate_id}={self.status} without model "
                f"evidence; pass evidence={{'model_id': ...}} from a real run or "
                f"call with write=False for smoke runs."
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cp = checkpoint_dir / f"{self.gate_id}.md"
        cp.write_text(self.to_markdown())
        # update status.json
        status = {}
        if status_path.exists():
            status = json.loads(status_path.read_text())
        status.setdefault("gates", {})
        status["gates"][self.gate_id] = self.status
        status["updated"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2))
        return {"checkpoint": str(cp), "status": str(status_path)}


def run_gate_checks(gate_id: str, title: str, checks: List[Tuple[str, bool, str]],
                    config: Optional[Dict] = None, write: bool = True,
                    evidence: Optional[Dict] = None) -> GateStatus:
    """Convenience: build, resolve, and (optionally) persist a GateStatus.

    ``evidence`` must contain a ``model_id`` for the write to be accepted into the
    shared status.json (see GateStatus.write). Smoke/synthetic callers pass
    ``write=False``.
    """
    gs = GateStatus(gate_id=gate_id, title=title, config=config or {},
                    evidence=evidence or {})
    for name, passed, detail in checks:
        gs.add(name, passed, detail)
    gs.resolve()
    if write:
        gs.write()
    return gs

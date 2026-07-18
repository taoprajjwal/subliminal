"""Run IDs, content hashing, and manifests.

Every expensive artifact must point to a JSON manifest (EXPERIMENT_PLAN.md §15).
This module has no heavy dependencies; git/torch/device probes are best-effort.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys)."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_info(repo: str | os.PathLike | None = None) -> Dict[str, Optional[str]]:
    """Best-effort git commit + dirty-diff hash. Returns Nones if not a repo."""
    repo = str(repo or Path(__file__).resolve().parents[2])
    info: Dict[str, Optional[str]] = {"commit": None, "uncommitted_diff_hash": None, "dirty": None}

    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", repo, *args],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                return None
            return out.stdout.strip()
        except Exception:
            return None

    info["commit"] = _run(["rev-parse", "HEAD"])
    diff = _run(["diff", "HEAD"])
    if diff is not None:
        info["dirty"] = bool(diff)
        info["uncommitted_diff_hash"] = sha256_text(diff) if diff else None
    return info


def device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def make_run_id(model_tag: str, target: str, phase: str, seed: int,
                git_sha: Optional[str] = None) -> str:
    """YYYYMMDD_HHMMSS_model_target_phase_gitsha_seed (EXPERIMENT_PLAN.md §15.1)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sha = (git_sha or git_info().get("commit") or "nogit")[:8]
    safe = lambda s: str(s).replace("/", "-").replace(" ", "")
    return f"{ts}_{safe(model_tag)}_{safe(target)}_{safe(phase)}_{sha}_{seed}"


@dataclass
class Manifest:
    """Provenance record for one artifact / run."""

    run_id: str
    phase: str
    parent_run_ids: List[str] = field(default_factory=list)
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    chat_template_hash: Optional[str] = None
    model_config_hash: Optional[str] = None
    attn_implementation: Optional[str] = None
    dtype: Optional[str] = None
    seed: Optional[int] = None
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    split_hashes: Dict[str, str] = field(default_factory=dict)
    prompt_hashes: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    command_line: Optional[str] = None
    package_lock_hash: Optional[str] = None
    code_commit: Optional[str] = None
    uncommitted_diff_hash: Optional[str] = None
    device: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, phase: str, model_tag: str = "na", target: str = "na",
               seed: int = 0, **kwargs: Any) -> "Manifest":
        g = git_info()
        return cls(
            run_id=make_run_id(model_tag, target, phase, seed, g.get("commit")),
            phase=phase,
            seed=seed,
            code_commit=g.get("commit"),
            uncommitted_diff_hash=g.get("uncommitted_diff_hash"),
            device=device_info(),
            start_time=utc_now_iso(),
            command_line=" ".join(__import__("sys").argv),
            **kwargs,
        )

    def finalize(self) -> "Manifest":
        self.end_time = utc_now_iso()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Manifest":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def references_hash(self, target_hash: str) -> bool:
        """True if target_hash appears anywhere in split/prompt hashes or sources.

        Used by the final evaluator's leakage guard (EXPERIMENT_PLAN.md §2.11):
        the final evaluator must refuse to run if a final-eval file hash appears
        in a search manifest.
        """
        blob = json.dumps(self.to_dict(), default=str)
        return target_hash in blob

"""Configuration loading.

Configs are plain YAML with an optional ``extends:`` key for single inheritance.
We deliberately use dataclasses + dicts (not pydantic) so this module imports
with only stdlib + pyyaml. A best-effort ``.env`` loader is included.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_dotenv(path: str | os.PathLike | None = None) -> Dict[str, str]:
    """Minimal .env loader (KEY=VALUE lines). Does not overwrite existing env."""
    path = Path(path) if path else REPO_ROOT / ".env"
    loaded: Dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config, resolving a single ``extends:`` chain relative to it."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        # try, in order: repo root, repo configs dir, configs dir without prefix
        candidates = [
            REPO_ROOT / path,
            CONFIGS_DIR / path,
            CONFIGS_DIR / Path(*path.parts[1:]) if path.parts[:1] == ("configs",) else None,
        ]
        for cand in candidates:
            if cand is not None and cand.exists():
                path = cand
                break
    data = yaml.safe_load(path.read_text()) or {}
    parent_ref = data.pop("extends", None)
    if parent_ref:
        parent_path = (path.parent / parent_ref).resolve()
        parent = load_yaml(parent_path)
        data = _deep_merge(parent, data)
    return data


# ----- execution-mode flags (env-driven; §14.3 of the plan) -----

def _envflag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RunModes:
    fast_dev_run: bool = True
    run_expensive: bool = False
    recompute: bool = False

    @classmethod
    def from_env(cls) -> "RunModes":
        return cls(
            fast_dev_run=_envflag("FAST_DEV_RUN", True),
            run_expensive=_envflag("RUN_EXPENSIVE", False),
            recompute=_envflag("RECOMPUTE", False),
        )


@dataclass
class ProjectPaths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    artifacts: Path = REPO_ROOT / "artifacts"
    reports: Path = REPO_ROOT / "reports"

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        data = Path(os.environ.get("SUBLIMINAL_DATA_ROOT", str(REPO_ROOT / "data")))
        art = Path(os.environ.get("SUBLIMINAL_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts")))
        return cls(root=REPO_ROOT, data=data, artifacts=art, reports=REPO_ROOT / "reports")


@dataclass
class Config:
    """A thin wrapper around a resolved config dict with convenient access."""

    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None
    modes: RunModes = field(default_factory=RunModes.from_env)
    paths: ProjectPaths = field(default_factory=ProjectPaths.from_env)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        load_dotenv()
        raw = load_yaml(path)
        return cls(raw=raw, source_path=Path(path))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

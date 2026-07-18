"""Activation caching (EXPERIMENT_PLAN.md §15.3).

Stores only required layers/positions/components. Uses float32 for aggregate
scores even when model activations are bfloat16. Backend is chunked ``.npz`` +
a JSON index by default (zarr optional; not required in gpu2). Never overwrites
an existing cache unless ``recompute=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ActivationRecord:
    """One cached activation tensor with its coordinates."""

    key: str  # e.g. f"{condition}/{layer}/{component}/{position}"
    array: np.ndarray  # (n_prompts, d) float32
    prompt_ids: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


class ActivationCache:
    """Directory-backed cache: <root>/<name>.npz + <root>/<name>.index.json."""

    def __init__(self, root: str, recompute: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.recompute = recompute

    def _paths(self, name: str):
        return self.root / f"{name}.npz", self.root / f"{name}.index.json"

    def exists(self, name: str) -> bool:
        npz, idx = self._paths(name)
        return npz.exists() and idx.exists()

    def save(self, name: str, records: Dict[str, ActivationRecord],
             manifest: Optional[dict] = None) -> Path:
        npz, idx = self._paths(name)
        if self.exists(name) and not self.recompute:
            raise FileExistsError(
                f"activation cache {name!r} exists; set recompute=True to overwrite"
            )
        arrays = {k: np.asarray(r.array, dtype=np.float32) for k, r in records.items()}
        np.savez_compressed(npz, **arrays)
        index = {
            "keys": {k: {"shape": list(arrays[k].shape),
                          "prompt_ids": r.prompt_ids, "meta": r.meta}
                      for k, r in records.items()},
            "manifest": manifest or {},
        }
        idx.write_text(json.dumps(index, indent=2, default=str))
        return npz

    def load(self, name: str) -> Dict[str, ActivationRecord]:
        npz, idx = self._paths(name)
        if not self.exists(name):
            raise FileNotFoundError(name)
        data = np.load(npz)
        index = json.loads(idx.read_text())
        out: Dict[str, ActivationRecord] = {}
        for k in data.files:
            info = index["keys"].get(k, {})
            out[k] = ActivationRecord(
                key=k, array=data[k], prompt_ids=info.get("prompt_ids", []),
                meta=info.get("meta", {}),
            )
        return out

    def load_index(self, name: str) -> dict:
        _, idx = self._paths(name)
        return json.loads(idx.read_text())

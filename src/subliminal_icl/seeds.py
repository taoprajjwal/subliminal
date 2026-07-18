"""Deterministic seeding utilities.

Seeds python ``random``, numpy, and (if present) torch. Also exposes helpers to
derive reproducible sub-seeds from a base seed + string label, so different
phases/objects get distinct-but-deterministic streams.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass


def _hash_to_uint32(*parts: object) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def derive_seed(base_seed: int, *labels: object) -> int:
    """Reproducibly derive a 32-bit sub-seed from a base seed and labels."""
    return (_hash_to_uint32(base_seed, *labels)) & 0x7FFFFFFF


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed python/numpy/torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:  # pragma: no cover - numpy always present in this project
        pass
    try:  # torch is optional at import time
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Best-effort determinism; some ops have no deterministic kernel.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed


@dataclass(frozen=True)
class SeedBundle:
    """A base seed plus lazily-derived named sub-seeds."""

    base: int

    def sub(self, *labels: object) -> int:
        return derive_seed(self.base, *labels)

    def numpy_rng(self, *labels: object):
        import numpy as np

        return np.random.default_rng(self.sub(*labels))

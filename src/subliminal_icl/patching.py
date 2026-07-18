"""Causal patching / ablation (EXPERIMENT_PLAN.md §5, §9, notebook 05).

Numpy-level operations (unit-testable) plus torch hook builders that apply them
during a forward pass. Interventions (with projector ``P_U`` onto the subspace):

    patch-in : h_patch   = h_0 + alpha * P_U (h_T - h_0)
    ablate   : h_ablated = h_T - P_U (h_T - h_0)

Controls: neutral->neutral, target->target, random/countertrait/sign-reversed/
norm-matched-orthogonal subspaces are produced in trait_subspace + here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


# --------------------------------------------------------------------------
# numpy core (testable without torch)
# --------------------------------------------------------------------------

def project(P: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v @ P.T


def patch_in(h0: np.ndarray, hT: np.ndarray, P: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """h0 + alpha * P (hT - h0)."""
    return np.asarray(h0) + alpha * project(P, np.asarray(hT) - np.asarray(h0))


def ablate(hT: np.ndarray, h0: np.ndarray, P: np.ndarray) -> np.ndarray:
    """hT - P (hT - h0): remove the subspace component of the (target-neutral) shift."""
    return np.asarray(hT) - project(P, np.asarray(hT) - np.asarray(h0))


def dose_response(h0: np.ndarray, hT: np.ndarray, P: np.ndarray,
                  alphas) -> Dict[float, np.ndarray]:
    return {float(a): patch_in(h0, hT, P, a) for a in alphas}


# --------------------------------------------------------------------------
# torch hook builders (used with hooks.patch_activations)
# --------------------------------------------------------------------------

def make_add_direction_patch(direction, alpha: float, position_index: Optional[int] = None):
    """Return a patch_fn(hidden, path) that adds ``alpha * direction`` at a token
    position (or all positions). ``direction`` is a 1-D torch tensor / array of
    size hidden_size. hidden is (batch, seq, hidden)."""
    def patch_fn(hidden, _path):
        import torch
        d = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
        new = hidden.clone()
        if position_index is None:
            new = new + alpha * d
        else:
            new[:, position_index, :] = new[:, position_index, :] + alpha * d
        return new
    return patch_fn


def make_projector_patch(projector_matrix, source_hidden, position_index: Optional[int] = None,
                         mode: str = "patch_in", alpha: float = 1.0):
    """Return a patch_fn implementing patch-in or ablation using a projector.

    ``source_hidden`` supplies h_T (patch_in) or the neutral reference (ablate)
    at the patched position; shape broadcastable to the hidden slice.
    """
    def patch_fn(hidden, _path):
        import torch
        P = torch.as_tensor(projector_matrix, dtype=hidden.dtype, device=hidden.device)
        src = torch.as_tensor(source_hidden, dtype=hidden.dtype, device=hidden.device)
        new = hidden.clone()

        def apply(slice_):
            delta = (src - slice_) @ P.T
            if mode == "patch_in":
                return slice_ + alpha * delta
            elif mode == "ablate":
                return slice_ - (slice_ - src) @ P.T
            raise ValueError(mode)

        if position_index is None:
            new = apply(new)
        else:
            new[:, position_index, :] = apply(new[:, position_index, :])
        return new
    return patch_fn


@dataclass
class InterventionResult:
    metric_baseline: float
    metric_intervened: float

    @property
    def delta(self) -> float:
        return self.metric_intervened - self.metric_baseline

"""Storage / retrieval tracing (EXPERIMENT_PLAN.md §9.3, notebook 06).

The retrieval score estimates how much target-aligned state written at earlier
source positions reaches a later query position. Two estimators:

  - attention-weighted head-value estimator (diagnostic only until confirmed by
    causal head patching);
  - causal source-position patch delta (patch a source-position activation from a
    high-score context into a neutral context and read the query-position score
    change) — the trustworthy signal per the plan.

Numpy core here; the causal-delta path is driven by hooks.patch_activations in
the notebooks/scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def attention_weighted_retrieval(attn_weights: np.ndarray, head_value_writes: np.ndarray,
                                 v_target: np.ndarray, query_index: int,
                                 source_indices: Sequence[int]) -> float:
    """sum over heads & source positions of
       attn_weight(query, source) * <v_target, head_value_write(source)>.

    ``attn_weights``: (n_heads, seq, seq); ``head_value_writes``: (n_heads, seq, d)
    the per-head value contribution written at each source position (already
    projected to model dim); ``v_target``: (d,).
    """
    aw = np.asarray(attn_weights, dtype=np.float64)
    hv = np.asarray(head_value_writes, dtype=np.float64)
    v = np.asarray(v_target, dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    total = 0.0
    for h in range(aw.shape[0]):
        for s in source_indices:
            total += float(aw[h, query_index, s]) * float(hv[h, s] @ v)
    return total


@dataclass
class RetrievalScore:
    attention_estimate: Optional[float] = None
    causal_delta: Optional[float] = None

    def value(self) -> float:
        """Prefer the causally-confirmed delta; fall back to the attention est."""
        return self.causal_delta if self.causal_delta is not None else (self.attention_estimate or 0.0)


ANCHOR_KINDS = (
    "assistant_start",
    "end_of_demonstration",
    "neutral_delimiter",
    "final_numeric_token",
    "synthetic_carrier_delimiter",
)


def check_attention_rows_normalized(attn_weights: np.ndarray, tol: float = 1e-3) -> bool:
    """Attention rows must sum to 1 (within tol) under eager attention."""
    aw = np.asarray(attn_weights, dtype=np.float64)
    sums = aw.sum(axis=-1)
    return bool(np.all(np.abs(sums - 1.0) <= tol))

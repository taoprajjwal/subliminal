"""Candidate-row activation scoring (EXPERIMENT_PLAN.md §9, notebook 07).

    score(d_i) = state_score
                 + rho * retrieval_score
                 - lambda_var * robustness_penalty
                 - lambda_len * length_penalty
                 - leakage_penalty        (infinite for any semantic hit)

The score NEVER touches the favorite-animal evaluator or target output logits.
Coefficients are fixed using only activation-validation data.

The numpy-level ``aggregate_score`` is unit-testable; the model-driven state /
retrieval measurement lives in the extraction path (hooks + trait_subspace).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .semantic_filters import SemanticScanner


@dataclass
class ScoringWeights:
    rho: float = 1.0            # retrieval weight
    lambda_var: float = 0.5     # cross-query robustness penalty
    lambda_len: float = 0.01    # length penalty
    leakage_penalty: float = float("inf")


@dataclass
class CandidateScore:
    row_id: str
    state_score: float
    retrieval_score: float
    robustness_penalty: float
    length_penalty: float
    leakage_hit: bool
    total: float
    source_label: Optional[str] = None
    comparison: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "row_id": self.row_id, "state_score": self.state_score,
            "retrieval_score": self.retrieval_score,
            "robustness_penalty": self.robustness_penalty,
            "length_penalty": self.length_penalty, "leakage_hit": self.leakage_hit,
            "total": self.total, "source_label": self.source_label,
            "comparison": self.comparison,
        }


def robustness_penalty_from_scores(per_query_scores: Sequence[float]) -> float:
    """Variance across neutral diagnostic queries."""
    if len(per_query_scores) < 2:
        return 0.0
    return float(np.var(per_query_scores, ddof=1))


def length_penalty_from_tokens(n_tokens: int, reference: int = 64) -> float:
    """Small penalty growing with |len - reference| (matched-stratum norm can
    replace this)."""
    return abs(n_tokens - reference) / max(reference, 1)


def aggregate_score(state_score: float, retrieval_score: float,
                    per_query_scores: Sequence[float], n_tokens: int,
                    leakage_hit: bool, weights: ScoringWeights,
                    row_id: str = "", source_label: Optional[str] = None,
                    comparison: Optional[Dict[str, float]] = None) -> CandidateScore:
    rp = robustness_penalty_from_scores(per_query_scores)
    lp = length_penalty_from_tokens(n_tokens)
    if leakage_hit:
        total = -weights.leakage_penalty
    else:
        total = (state_score + weights.rho * retrieval_score
                 - weights.lambda_var * rp - weights.lambda_len * lp)
    return CandidateScore(
        row_id=row_id, state_score=float(state_score), retrieval_score=float(retrieval_score),
        robustness_penalty=rp, length_penalty=lp, leakage_hit=bool(leakage_hit),
        total=float(total), source_label=source_label, comparison=comparison or {},
    )


def leakage_hit_for_text(text: str, target: str, scanner: Optional[SemanticScanner] = None) -> bool:
    scanner = scanner or SemanticScanner()
    return not scanner.is_strict_clean(text, targets=[target])

"""Activation-only candidate scoring + context search on a live model (Gates 4B/5).

For a candidate demonstration ``d`` and neutral diagnostic queries ``q``, we read
the frozen model on the concatenated text [d]\n[q] and measure the trait-subspace
signed score at two positions:

  state_score     : subspace score at the end-of-demonstration token (writing)
  retrieval_score : subspace score at the final query token (reaching the query)

Neither uses the favorite-animal evaluator (EXPERIMENT_PLAN.md §9, notebook 07/08).
The beam-search ``score_fn`` here is what search_contexts.py wraps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .chat_templates import SystemMode, render_chat
from .model_adapters import residual_module_path
from .candidate_scoring import ScoringWeights, aggregate_score, CandidateScore, leakage_hit_for_text


def _residual_last(loaded, text: str, layer: int) -> np.ndarray:
    """Return the residual-stream vector at the final token of ``text``."""
    import torch
    from .hooks import capture_activations

    model, tok = loaded.model, loaded.tokenizer
    rc = render_chat(text, SystemMode.NONE, tokenizer=tok)
    ids = list(rc.token_ids)
    input_ids = torch.tensor([ids], device=next(model.parameters()).device)
    attn = torch.ones_like(input_ids)
    path = residual_module_path(model, layer)
    with capture_activations(model, [path], detach=True, to_cpu=True) as store:
        with torch.no_grad():
            model(input_ids, attention_mask=attn)
    h = store.get(path)[0]  # (seq, hidden)
    return h[-1].to(torch.float32).numpy()


@dataclass
class ContextScore:
    state_score: float
    retrieval_score: float
    per_query_state: List[float]


def score_context(loaded, subspace, demo_text: str, diagnostic_queries: Sequence[str],
                  delimiter: str = "\n") -> ContextScore:
    """Measure state + retrieval subspace scores of a demonstration/context.

    ``demo_text`` may be a single row or several rows already concatenated.
    """
    # write score: final token of the demo/context rendered alone
    demo_vec = _residual_last(loaded, demo_text, subspace.layer)
    state = subspace.signed_score(demo_vec)
    # retrieval score: final token of [demo + query], averaged over queries
    retrievals = []
    for q in diagnostic_queries:
        vec = _residual_last(loaded, demo_text + delimiter + q, subspace.layer)
        retrievals.append(subspace.signed_score(vec))
    return ContextScore(state_score=float(state),
                        retrieval_score=float(np.mean(retrievals)),
                        per_query_state=retrievals)


def score_candidate_rows(loaded, subspace, rows: Sequence[dict], target: str,
                         diagnostic_queries: Sequence[str],
                         weights: Optional[ScoringWeights] = None) -> List[CandidateScore]:
    """Score each candidate row (dict with 'row_id','assistant_text',...)."""
    weights = weights or ScoringWeights()
    out: List[CandidateScore] = []
    for r in rows:
        text = r.get("assistant_text", "")
        cs = score_context(loaded, subspace, text, diagnostic_queries)
        n_tokens = len(loaded.tokenizer(text, add_special_tokens=False)["input_ids"])
        leak = leakage_hit_for_text(text, target)
        out.append(aggregate_score(
            cs.state_score, cs.retrieval_score, cs.per_query_state, n_tokens, leak,
            weights, row_id=r.get("row_id", ""), source_label=r.get("source_label")))
    return out


def make_context_score_fn(loaded, subspace, rows_by_id: Dict[str, dict],
                          diagnostic_queries: Sequence[str],
                          weights: Optional[ScoringWeights] = None) -> Callable[[List[str]], float]:
    """Build a beam-search score_fn over row ids (activation-only; no evaluator)."""
    weights = weights or ScoringWeights()

    def score_fn(item_ids: List[str]) -> float:
        text = "\n".join(rows_by_id[i].get("assistant_text", "") for i in item_ids)
        cs = score_context(loaded, subspace, text, diagnostic_queries)
        # constructive objective: writing + retrieval reaching the query
        return cs.state_score + weights.rho * cs.retrieval_score

    return score_fn

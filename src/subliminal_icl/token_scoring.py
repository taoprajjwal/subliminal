"""Token scoring (EXPERIMENT_PLAN.md §6.3).

Never first-token-only: candidate answers are scored by the full-sequence
conditional log-probability of every candidate token. The numerical core works
on plain arrays so it is unit-testable without a model; the model-facing helpers
lazily import torch/transformers and feed the same core.

Primary metric:
    F_T = log p(target answer | prompt)
          - logmeanexp_{j != T} log p(other animal j | prompt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    z = logits - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def logmeanexp(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m))))


def token_logprobs_manual(logits: np.ndarray, target_ids: Sequence[int]) -> List[float]:
    """Reference implementation via an explicit per-position loop.

    ``logits`` has shape (T, V): row t are the next-token logits produced *after*
    consuming the t-th token, i.e. they predict token t+1 in a causal LM. This
    function returns, for each t in range(len(target_ids)), the log-prob assigned
    to ``target_ids[t]`` by row t. Callers align ``logits`` and ``target_ids`` so
    that row t predicts target token t (see ``candidate_logprob``).
    """
    logits = np.asarray(logits, dtype=np.float64)
    out: List[float] = []
    for t, tok in enumerate(target_ids):
        row = logits[t]
        m = row.max()
        lse = m + np.log(np.sum(np.exp(row - m)))
        out.append(float(row[tok] - lse))
    return out


def token_logprobs(logits: np.ndarray, target_ids: Sequence[int]) -> np.ndarray:
    """Vectorized equivalent of ``token_logprobs_manual``."""
    lp = log_softmax(np.asarray(logits, dtype=np.float64), axis=-1)
    idx = np.asarray(target_ids, dtype=np.int64)
    return lp[np.arange(len(idx)), idx]


def sequence_logprob(logits_for_candidate: np.ndarray,
                     candidate_ids: Sequence[int]) -> Dict[str, float]:
    """Sum + per-token-normalized log-prob of a candidate answer.

    ``logits_for_candidate`` is the (len(candidate_ids), V) slice of next-token
    logits aligned so row t predicts candidate token t.
    """
    per = token_logprobs(logits_for_candidate, candidate_ids)
    total = float(np.sum(per))
    n = max(len(candidate_ids), 1)
    return {"total_logprob": total, "per_token_logprob": total / n, "n_tokens": len(candidate_ids)}


def target_margin(candidate_logprobs: Dict[str, float], target: str) -> float:
    """F_T from a map {animal: total_logprob}."""
    if target not in candidate_logprobs:
        raise KeyError(f"target {target!r} not scored")
    others = [v for k, v in candidate_logprobs.items() if k != target]
    return float(candidate_logprobs[target] - logmeanexp(others))


def top1_rate(candidate_logprobs: Dict[str, float], target: str) -> int:
    best = max(candidate_logprobs, key=candidate_logprobs.get)
    return int(best == target)


# --------------------------------------------------------------------------
# model-facing helpers (lazy torch/transformers import)
# --------------------------------------------------------------------------

@dataclass
class ScoredAnswer:
    text: str
    token_ids: List[int]
    total_logprob: float
    per_token_logprob: float


def candidate_logprob(model, tokenizer, prompt_ids: Sequence[int],
                      candidate_text: str, device: Optional[str] = None) -> ScoredAnswer:
    """Full-sequence conditional log-prob of ``candidate_text`` after ``prompt_ids``.

    ``prompt_ids`` must be the *identical* rendered answer-prefix token ids for
    every candidate (EXPERIMENT_PLAN.md §6.3 / test_answer_prefix_is_identical).
    """
    import torch  # lazy

    device = device or (next(model.parameters()).device.type if hasattr(model, "parameters") else "cpu")
    cand_ids = tokenizer(candidate_text, add_special_tokens=False)["input_ids"]
    full = list(prompt_ids) + list(cand_ids)
    input_ids = torch.tensor([full], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (T, V)
    # row t predicts token t+1; candidate token c (0-indexed within candidate)
    # is at absolute position len(prompt)+c, predicted by row len(prompt)+c-1.
    start = len(prompt_ids) - 1
    end = start + len(cand_ids)
    cand_logits = logits[start:end].to(torch.float32).cpu().numpy()
    res = sequence_logprob(cand_logits, cand_ids)
    return ScoredAnswer(candidate_text, cand_ids, res["total_logprob"], res["per_token_logprob"])

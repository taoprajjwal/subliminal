"""Behavioral evaluation of animal preference (EXPERIMENT_PLAN.md §6.3, notebook 09).

Scores every animal candidate by full-sequence log-prob after an identical
answer prefix, computes the target margin F_T and the full confusion matrix, and
tracks the ``Qwen`` model-name artifact. Model-facing; torch/transformers lazy.
The numpy-level confusion-matrix + diagonal assembly is testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ANIMALS
from .chat_templates import SystemMode, render_chat
from .token_scoring import candidate_logprob, target_margin, top1_rate


@dataclass
class EvalItem:
    prompt: str
    family: str
    system_mode: str = SystemMode.NONE.value


@dataclass
class EvalResult:
    prompt: str
    family: str
    logprobs: Dict[str, float]           # animal -> total logprob
    target_margin: float
    top1_target: int
    qwen_logprob: Optional[float] = None

    def to_dict(self):
        return {
            "prompt": self.prompt, "family": self.family, "logprobs": self.logprobs,
            "target_margin": self.target_margin, "top1_target": self.top1_target,
            "qwen_logprob": self.qwen_logprob,
        }


def score_prompt(model, tokenizer, context_text: str, question: str, target: str,
                 animals: Sequence[str] = ANIMALS, family: str = "direct",
                 answer_prefix: str = "", track_qwen: bool = True) -> EvalResult:
    """Score all animal candidates for one (context + question).

    The context is prepended as plain text; the *identical* answer prefix token
    ids are used for every candidate (§6.3).
    """
    user = (context_text + "\n\n" + question) if context_text else question
    rc = render_chat(user, SystemMode.NONE, tokenizer=tokenizer)
    prefix_ids = list(rc.token_ids or [])
    if answer_prefix:
        prefix_ids += tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]

    logprobs: Dict[str, float] = {}
    for a in animals:
        scored = candidate_logprob(model, tokenizer, prefix_ids, a)
        logprobs[a] = scored.total_logprob
    qwen_lp = None
    if track_qwen:
        qwen_lp = candidate_logprob(model, tokenizer, prefix_ids, "Qwen").total_logprob
    return EvalResult(
        prompt=question, family=family, logprobs=logprobs,
        target_margin=target_margin(logprobs, target),
        top1_target=top1_rate(logprobs, target), qwen_logprob=qwen_lp,
    )


# --------------------------------------------------------------------------
# aggregation (numpy-only, testable)
# --------------------------------------------------------------------------

def confusion_matrix(results_by_target: Dict[str, List[EvalResult]],
                     animals: Sequence[str] = ANIMALS) -> np.ndarray:
    """Rows = target context trait, cols = measured animal; entry = mean logprob."""
    animals = list(animals)
    M = np.full((len(animals), len(animals)), np.nan)
    for i, t in enumerate(animals):
        res = results_by_target.get(t)
        if not res:
            continue
        for j, a in enumerate(animals):
            vals = [r.logprobs.get(a, np.nan) for r in res]
            M[i, j] = float(np.nanmean(vals))
    return M


def diagonal_minus_offdiagonal(M: np.ndarray) -> float:
    """Mean diagonal minus mean off-diagonal of a confusion matrix — a quick
    target-specificity summary (the rigorous version is statistics.gamma)."""
    d = np.nanmean(np.diag(M))
    mask = ~np.eye(M.shape[0], dtype=bool)
    off = np.nanmean(M[mask])
    return float(d - off)

"""Selection-only context beam search (EXPERIMENT_PLAN.md §9.4, notebook 08).

Generic and model-agnostic: the caller supplies a ``score_fn`` that maps a
context (list of item ids) to a scalar activation-only score, and a
``propose_fn`` that yields candidate next-item ids for a beam. The scorer MUST
NOT touch the behavioral evaluator (enforced by the clean-import guard in the
search script, not here). Search state is fully serializable so it resumes
exactly from a checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Beam:
    items: List[str]
    score: float
    trajectory: List[float] = field(default_factory=list)  # score after each step
    meta: Dict[str, object] = field(default_factory=dict)

    def clone_with(self, item: str, score: float) -> "Beam":
        return Beam(
            items=self.items + [item],
            score=score,
            trajectory=self.trajectory + [score],
            meta=dict(self.meta),
        )


@dataclass
class DiversityConstraints:
    """Constraints preventing trivial near-duplicate contexts (§8, notebook 08)."""

    forbid_duplicate_ids: bool = True
    forbid_duplicate_source_prompt: bool = True
    max_from_one_source: Optional[int] = None
    ngram_block: int = 0  # >0: block repeated n-grams of item source-prompt ids

    def violates(self, items: Sequence[str], source_of: Dict[str, str]) -> bool:
        if self.forbid_duplicate_ids and len(set(items)) != len(items):
            return True
        if self.forbid_duplicate_source_prompt:
            srcs = [source_of.get(i) for i in items if source_of.get(i) is not None]
            if len(set(srcs)) != len(srcs):
                return True
        if self.max_from_one_source is not None and source_of:
            from collections import Counter
            c = Counter(source_of.get(i) for i in items)
            if max(c.values()) > self.max_from_one_source:
                return True
        return False


ScoreFn = Callable[[List[str]], float]
ProposeFn = Callable[[Beam], Sequence[str]]


@dataclass
class SearchConfig:
    beam_width: int = 16
    proposal_count: int = 128
    max_steps: int = 16
    seed: int = 0
    diversity_penalty: float = 0.0  # subtracted per shared-source overlap in top-k selection


def _diverse_top_k(proposals: List[Beam], k: int, penalty: float) -> List[Beam]:
    """Greedy diversity-aware selection: pick highest score, then penalize
    proposals sharing items with already-selected beams."""
    if penalty <= 0:
        return sorted(proposals, key=lambda b: b.score, reverse=True)[:k]
    remaining = sorted(proposals, key=lambda b: b.score, reverse=True)
    selected: List[Beam] = []
    chosen_items: set = set()
    while remaining and len(selected) < k:
        # recompute penalized score
        best_i, best_val = 0, -np.inf
        for i, b in enumerate(remaining):
            overlap = len(set(b.items) & chosen_items)
            val = b.score - penalty * overlap
            if val > best_val:
                best_val, best_i = val, i
        pick = remaining.pop(best_i)
        selected.append(pick)
        chosen_items |= set(pick.items)
    return selected


@dataclass
class BeamSearchState:
    step: int
    beams: List[Beam]
    config: SearchConfig
    all_beams_by_step: List[List[List[str]]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "step": self.step,
            "config": asdict(self.config),
            "beams": [asdict(b) for b in self.beams],
            "all_beams_by_step": self.all_beams_by_step,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "BeamSearchState":
        d = json.loads(s)
        cfg = SearchConfig(**d["config"])
        beams = [Beam(**b) for b in d["beams"]]
        st = cls(step=d["step"], beams=beams, config=cfg,
                 all_beams_by_step=d.get("all_beams_by_step", []))
        return st

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "BeamSearchState":
        return cls.from_json(Path(path).read_text())


def beam_search(score_fn: ScoreFn, propose_fn: ProposeFn, config: SearchConfig,
                constraints: Optional[DiversityConstraints] = None,
                source_of: Optional[Dict[str, str]] = None,
                state: Optional[BeamSearchState] = None,
                checkpoint_cb: Optional[Callable[[BeamSearchState], None]] = None
                ) -> BeamSearchState:
    """Run (or resume) beam search. Saves all beams per step, not only winners."""
    constraints = constraints or DiversityConstraints()
    source_of = source_of or {}
    if state is None:
        state = BeamSearchState(step=0, beams=[Beam(items=[], score=0.0)], config=config)

    while state.step < config.max_steps:
        proposals: List[Beam] = []
        for beam in state.beams:
            for cand in propose_fn(beam)[: config.proposal_count]:
                items = beam.items + [cand]
                if constraints.violates(items, source_of):
                    continue
                sc = float(score_fn(items))
                proposals.append(beam.clone_with(cand, sc))
        if not proposals:
            break
        state.beams = _diverse_top_k(proposals, config.beam_width, config.diversity_penalty)
        state.all_beams_by_step.append([b.items for b in state.beams])
        state.step += 1
        if checkpoint_cb:
            checkpoint_cb(state)
    return state


def best_beam(state: BeamSearchState) -> Beam:
    return max(state.beams, key=lambda b: b.score)

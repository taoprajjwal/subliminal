"""Normalized data schemas (EXPERIMENT_PLAN.md §5.9).

Dataclass-based so the module imports with only stdlib. ``stable_row_id`` gives a
content hash used as the immutable ``row_id`` and for split-disjointness checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

TEACHER_CONDITIONS = ("neutral", "target", "countertrait", "negative_target", "random")
SPLITS = (
    "discovery",
    "prototype_train",
    "subspace_validation",
    "causal_patch_validation",
    "prototype_holdout",
    "candidate_search",
    "eval_dev",
    "eval_final",
)


def stable_row_id(source_dataset: str, source_index: Any, user_text: str,
                  assistant_text: str, teacher_condition: str = "") -> str:
    payload = "␟".join(
        [str(source_dataset), str(source_index), user_text, assistant_text, teacher_condition]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptCompletionRow:
    """One normalized prompt/completion row."""

    row_id: str
    source_dataset: str
    source_revision: Optional[str] = None
    source_index: Optional[int] = None
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    teacher_condition: str = "neutral"
    target_trait: Optional[str] = None
    countertrait: Optional[str] = None
    system_prompt_hash: Optional[str] = None
    user_text: str = ""
    assistant_text: str = ""
    rendered_chat_text: Optional[str] = None
    input_token_ids: List[int] = field(default_factory=list)
    assistant_token_ids: List[int] = field(default_factory=list)
    generation_seed: Optional[int] = None
    sampling: Sampling = field(default_factory=Sampling)
    divergence_mask: List[bool] = field(default_factory=list)
    semantic_filter_hits: List[Dict[str, Any]] = field(default_factory=list)
    split: Optional[str] = None
    source_label: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, source_dataset: str, user_text: str, assistant_text: str,
              teacher_condition: str = "neutral", source_index: Optional[int] = None,
              **kwargs: Any) -> "PromptCompletionRow":
        rid = stable_row_id(source_dataset, source_index, user_text, assistant_text,
                            teacher_condition)
        return cls(
            row_id=rid,
            source_dataset=source_dataset,
            source_index=source_index,
            user_text=user_text,
            assistant_text=assistant_text,
            teacher_condition=teacher_condition,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sampling"] = self.sampling.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptCompletionRow":
        d = dict(d)
        samp = d.get("sampling") or {}
        if isinstance(samp, dict):
            d["sampling"] = Sampling(**samp)
        # tolerate unknown keys by shoving them into extra
        known = set(cls.__dataclass_fields__.keys())
        extra = d.pop("extra", {}) or {}
        for k in list(d.keys()):
            if k not in known:
                extra[k] = d.pop(k)
        d["extra"] = extra
        return cls(**d)


def validate_row(row: PromptCompletionRow) -> List[str]:
    """Return a list of validation problems (empty == valid)."""
    problems: List[str] = []
    if row.teacher_condition not in TEACHER_CONDITIONS:
        problems.append(f"unknown teacher_condition={row.teacher_condition!r}")
    if row.split is not None and row.split not in SPLITS:
        problems.append(f"unknown split={row.split!r}")
    if row.divergence_mask and row.assistant_token_ids:
        if len(row.divergence_mask) != len(row.assistant_token_ids):
            problems.append(
                "divergence_mask length %d != assistant_token_ids length %d"
                % (len(row.divergence_mask), len(row.assistant_token_ids))
            )
    if not row.row_id:
        problems.append("empty row_id")
    return problems

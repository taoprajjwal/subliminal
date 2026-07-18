"""Dataset normalization, filtering, splitting, hashing (notebook 01, §5).

Normalizes arbitrary source rows into ``PromptCompletionRow`` (schema §5.9),
applies the semantic filter, and builds immutable, disjoint splits with stable
hashes. HF ``datasets`` is imported lazily; everything else is stdlib/numpy so
the split logic is unit-testable on in-memory fixtures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .data_schemas import PromptCompletionRow, SPLITS
from .manifests import sha256_json, sha256_text
from .semantic_filters import SemanticScanner

_NUMERIC_ROW = re.compile(r"^[\s0-9,;.\-]+$")


def is_numeric_output(text: str) -> bool:
    return bool(_NUMERIC_ROW.match(text.strip()))


def normalize_existing_eagle_row(index: int, row: Dict, source_dataset: str,
                                 source_revision: Optional[str] = None) -> PromptCompletionRow:
    """Adapt one row of the existing eagle-numbers dataset. The real column names
    are discovered at runtime (notebook 01 prints them); we map the common ones.
    """
    user = row.get("prompt") or row.get("question") or row.get("user") or row.get("input") or ""
    assistant = (row.get("completion") or row.get("response") or row.get("answer")
                 or row.get("output") or row.get("numbers") or "")
    return PromptCompletionRow.build(
        source_dataset=source_dataset, source_index=index,
        user_text=str(user), assistant_text=str(assistant),
        teacher_condition="target", target_trait="eagle",
        source_revision=source_revision, source_label="existing_eagle",
    )


def apply_semantic_filter(row: PromptCompletionRow, target: str,
                          scanner: Optional[SemanticScanner] = None) -> PromptCompletionRow:
    scanner = scanner or SemanticScanner()
    hits = scanner.scan(row.assistant_text + "\n" + row.user_text, targets=[target])
    row.semantic_filter_hits = [h.to_dict() for h in hits]
    return row


def filter_rows(rows: Iterable[PromptCompletionRow], target: str,
                require_numeric: bool = True,
                scanner: Optional[SemanticScanner] = None) -> Dict[str, List[PromptCompletionRow]]:
    """Split rows into kept / dropped. Dropping is independent of any activation
    objective (§5.2): only leakage + malformed numeric content drop a row."""
    scanner = scanner or SemanticScanner()
    kept, dropped = [], []
    for r in rows:
        r = apply_semantic_filter(r, target, scanner)
        strict_hit = any(h["match_level"] in {"exact", "normalized", "word"}
                         for h in r.semantic_filter_hits)
        malformed = require_numeric and not is_numeric_output(r.assistant_text)
        if strict_hit or malformed:
            dropped.append(r)
        else:
            kept.append(r)
    return {"kept": kept, "dropped": dropped}


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def assign_splits(rows: Sequence[PromptCompletionRow], sizes: Dict[str, int],
                  seed: int = 0, by_source_prompt: bool = True) -> Dict[str, List[PromptCompletionRow]]:
    """Deterministically assign rows to disjoint splits.

    If ``by_source_prompt`` is True, all rows sharing a ``source_index`` (source
    prompt) go to the same split, so no source prompt id is shared across train
    and final holdout (§10.1 test_split_disjointness).
    """
    rng = np.random.default_rng(seed)
    if by_source_prompt:
        groups: Dict[object, List[PromptCompletionRow]] = {}
        for r in rows:
            groups.setdefault(r.source_index, []).append(r)
        units = list(groups.values())
    else:
        units = [[r] for r in rows]
    order = rng.permutation(len(units))
    out: Dict[str, List[PromptCompletionRow]] = {s: [] for s in sizes}
    pos = 0
    for split, n in sizes.items():
        taken = 0
        while taken < n and pos < len(units):
            unit = units[order[pos]]
            for r in unit:
                r.split = split
                out[split].append(r)
            taken += 1
            pos += 1
    return out


def split_hash(rows: Sequence[PromptCompletionRow]) -> str:
    return sha256_json(sorted(r.row_id for r in rows))


def check_disjoint(splits: Dict[str, List[PromptCompletionRow]]) -> Dict[str, object]:
    """Verify no row_id and no source prompt id are shared across splits."""
    ids_by_split = {s: {r.row_id for r in rs} for s, rs in splits.items()}
    src_by_split = {s: {r.source_index for r in rs} for s, rs in splits.items()}
    problems = []
    keys = list(splits)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if ids_by_split[a] & ids_by_split[b]:
                problems.append(f"row_id overlap {a}&{b}")
            if src_by_split[a] & src_by_split[b]:
                problems.append(f"source_prompt overlap {a}&{b}")
    return {"disjoint": not problems, "problems": problems}


def write_split_jsonl(rows: Sequence[PromptCompletionRow], path: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(r.to_json() + "\n")
    return split_hash(rows)


def read_split_jsonl(path: str) -> List[PromptCompletionRow]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(PromptCompletionRow.from_dict(json.loads(line)))
    return out


def length_match_report(a_lens: Sequence[int], b_lens: Sequence[int]) -> Dict[str, float]:
    a, b = np.asarray(a_lens, float), np.asarray(b_lens, float)
    return {
        "mean_a": float(a.mean()) if a.size else float("nan"),
        "mean_b": float(b.mean()) if b.size else float("nan"),
        "mean_abs_diff": float(abs(a.mean() - b.mean())) if a.size and b.size else float("nan"),
    }

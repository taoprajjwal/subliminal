import numpy as np
import pytest

from subliminal_icl.data_schemas import PromptCompletionRow, stable_row_id, validate_row
from subliminal_icl import data_builders as db
from subliminal_icl.semantic_filters import SemanticScanner, normalize_text
from subliminal_icl.chat_templates import compare_system_modes, SystemMode, render_chat

pytestmark = pytest.mark.unit


# ---- data / splits ----

def test_stable_row_hashes():
    a = stable_row_id("ds", 1, "u", "a", "target")
    b = stable_row_id("ds", 1, "u", "a", "target")
    c = stable_row_id("ds", 2, "u", "a", "target")
    assert a == b and a != c


def test_split_disjointness_and_hash_stability():
    rows = [PromptCompletionRow.build("ds", f"seq {i}", "1, 2, 3", source_index=i)
            for i in range(12)]
    sizes = {"candidate_search": 3, "eval_dev": 1, "eval_final": 1}
    s1 = db.assign_splits(rows, sizes, seed=0)
    s2 = db.assign_splits(rows, sizes, seed=0)
    assert db.check_disjoint(s1)["disjoint"]
    assert db.split_hash(s1["candidate_search"]) == db.split_hash(s2["candidate_search"])


def test_by_source_prompt_keeps_groups_together():
    # two rows share source_index -> must land in same split
    rows = [PromptCompletionRow.build("ds", "u", "1,2", source_index=0),
            PromptCompletionRow.build("ds", "u2", "3,4", source_index=0),
            PromptCompletionRow.build("ds", "u3", "5,6", source_index=1)]
    splits = db.assign_splits(rows, {"a": 1, "b": 1}, seed=0, by_source_prompt=True)
    assert db.check_disjoint(splits)["disjoint"]


def test_divergence_mask_alignment_validation():
    r = PromptCompletionRow.build("ds", "u", "1,2", source_index=0)
    r.assistant_token_ids = [10, 11, 12]
    r.divergence_mask = [False, True]  # wrong length
    problems = validate_row(r)
    assert any("divergence_mask" in p for p in problems)


def test_row_roundtrip():
    r = PromptCompletionRow.build("ds", "u", "1,2,3", source_index=5, target_trait="eagle")
    d = r.to_dict()
    r2 = PromptCompletionRow.from_dict(d)
    assert r2.row_id == r.row_id and r2.target_trait == "eagle"


# ---- semantic leakage scanner ----

def test_semantic_leakage_scanner_exact_and_word():
    sc = SemanticScanner()
    assert not sc.is_strict_clean("I really love eagles", ["eagle"])
    assert sc.is_strict_clean("12, 45, 903, 7", ["eagle"])


def test_semantic_leakage_translation_and_plural():
    sc = SemanticScanner()
    hits = sc.scan("mi animal favorito es el aguila", ["eagle"])
    assert any(h.category == "translations" for h in hits)


def test_semantic_leakage_reports_not_deletes():
    sc = SemanticScanner()
    hits = sc.scan("eagles and wolves", ["eagle", "wolf"])
    animals = {h.animal for h in hits}
    assert "eagle" in animals and "wolf" in animals


def test_normalize_text_strips_accents():
    assert normalize_text("águila") == "aguila"


def test_filter_rows_drops_leaky_and_malformed():
    rows = [
        PromptCompletionRow.build("ds", "seq", "1, 2, 3", source_index=0),
        PromptCompletionRow.build("ds", "seq", "eagles!", source_index=1),
        PromptCompletionRow.build("ds", "seq", "not numbers here", source_index=2),
    ]
    res = db.filter_rows(rows, "eagle")
    assert len(res["kept"]) == 1
    assert len(res["dropped"]) == 2


# ---- chat templates ----

def test_no_system_vs_empty_system_logged_separately():
    modes = compare_system_modes("Hello")
    assert modes[SystemMode.NONE.value].text != modes[SystemMode.EMPTY.value].text


def test_rendered_prompt_hash_stability():
    a = render_chat("hi", SystemMode.NONE)
    b = render_chat("hi", SystemMode.NONE)
    assert a.text == b.text and a.template_hash == b.template_hash


def test_length_match_report():
    rep = db.length_match_report([10, 12, 14], [11, 13, 15])
    assert rep["mean_abs_diff"] == pytest.approx(1.0)

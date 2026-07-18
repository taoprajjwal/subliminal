import json

import numpy as np
import pytest

from subliminal_icl import seeds
from subliminal_icl.manifests import (
    Manifest, make_run_id, sha256_text, sha256_json, sha256_file,
)

pytestmark = pytest.mark.unit


def test_derive_seed_deterministic_and_distinct():
    assert seeds.derive_seed(0, "a") == seeds.derive_seed(0, "a")
    assert seeds.derive_seed(0, "a") != seeds.derive_seed(0, "b")
    assert seeds.derive_seed(1, "a") != seeds.derive_seed(0, "a")


def test_seed_everything_reproducible_numpy_stream():
    seeds.seed_everything(123)
    a = np.random.rand(5)
    seeds.seed_everything(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_seed_bundle_rng_reproducible():
    sb = seeds.SeedBundle(7)
    r1 = sb.numpy_rng("x").standard_normal(4)
    r2 = sb.numpy_rng("x").standard_normal(4)
    assert np.allclose(r1, r2)


def test_run_id_format():
    rid = make_run_id("qwen7b", "eagle", "pilot", 0, git_sha="abcdef123456")
    parts = rid.split("_")
    assert parts[-1] == "0"
    assert "qwen7b" in rid and "eagle" in rid and "pilot" in rid


def test_manifest_save_load_and_leakage_guard(tmp_path):
    man = Manifest.create(phase="search", model_tag="qwen7b", target="eagle", seed=0)
    man.split_hashes["candidate_search"] = "deadbeef"
    p = tmp_path / "m.json"
    man.finalize().save(p)
    loaded = Manifest.load(p)
    assert loaded.phase == "search"
    # leakage guard: a final-eval hash appearing in a search manifest is detectable
    assert loaded.references_hash("deadbeef")
    assert not loaded.references_hash("not-present-hash")


def test_content_hashes_stable():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_sha256_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert sha256_file(p) == sha256_text("hello")

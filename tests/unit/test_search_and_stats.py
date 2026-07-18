import numpy as np
import pytest

from subliminal_icl import beam_search as bs
from subliminal_icl import statistics as st
from subliminal_icl import cot_carriers as cc

pytestmark = pytest.mark.unit


# ---- beam search ----

def _additive_problem(seed=0):
    rng = np.random.default_rng(seed)
    pool = {f"r{i}": float(rng.normal(0, 1)) for i in range(20)}
    src = {r: f"s{int(r[1:]) % 5}" for r in pool}
    score = lambda items: float(sum(pool[i] for i in items))
    propose = lambda beam: list(pool.keys())
    return pool, src, score, propose


def test_search_reproducibility():
    _, src, score, propose = _additive_problem()
    cfg = bs.SearchConfig(beam_width=5, proposal_count=20, max_steps=3)
    a = bs.beam_search(score, propose, cfg, source_of=src)
    b = bs.beam_search(score, propose, cfg, source_of=src)
    assert [x.items for x in a.beams] == [x.items for x in b.beams]


def test_no_duplicate_rows_in_context():
    _, src, score, propose = _additive_problem()
    cfg = bs.SearchConfig(beam_width=5, proposal_count=20, max_steps=4)
    st_ = bs.beam_search(score, propose, cfg, source_of=src)
    for beam in st_.beams:
        assert len(set(beam.items)) == len(beam.items)


def test_no_duplicate_source_prompt():
    _, src, score, propose = _additive_problem()
    cons = bs.DiversityConstraints(forbid_duplicate_source_prompt=True)
    cfg = bs.SearchConfig(beam_width=5, proposal_count=20, max_steps=3)
    st_ = bs.beam_search(score, propose, cfg, constraints=cons, source_of=src)
    for beam in st_.beams:
        srcs = [src[i] for i in beam.items]
        assert len(set(srcs)) == len(srcs)


def test_resume_from_checkpoint(tmp_path):
    _, src, score, propose = _additive_problem()
    cfg = bs.SearchConfig(beam_width=5, proposal_count=20, max_steps=3)
    full = bs.beam_search(score, propose, cfg, source_of=src)
    p = tmp_path / "state.json"
    full.save(str(p))
    loaded = bs.BeamSearchState.load(str(p))
    assert [b.items for b in loaded.beams] == [b.items for b in full.beams]
    assert loaded.step == full.step


def test_max_from_one_source_constraint():
    pool = {f"r{i}": 1.0 for i in range(10)}
    src = {r: "only_source" for r in pool}
    cons = bs.DiversityConstraints(forbid_duplicate_source_prompt=False, max_from_one_source=2)
    cfg = bs.SearchConfig(beam_width=3, proposal_count=10, max_steps=4)
    st_ = bs.beam_search(lambda it: float(len(it)), lambda b: list(pool),
                         cfg, constraints=cons, source_of=src)
    for beam in st_.beams:
        assert len(beam.items) <= 2


# ---- statistics ----

def test_diagonal_model_recovers_gamma():
    t, j, q, d = [], [], [], []
    rng = np.random.default_rng(0)
    animals = [f"a{i}" for i in range(5)]
    for tt in animals:
        for jj in animals:
            for qi in range(6):
                t.append(tt); j.append(jj); q.append(qi)
                d.append((1.0 if tt == jj else 0.0) + rng.normal(0, 0.1))
    res = st.fit_diagonal_model(st.DiagonalObservations(t=t, j=j, q=q, delta=d), n_boot=300)
    assert res["gamma"] == pytest.approx(1.0, abs=0.15)
    assert res["gamma_excludes_zero"]


def test_permutation_test_budget_matched():
    rng = np.random.default_rng(1)
    null_max = rng.normal(0, 0.3, size=100)
    res = st.permutation_test(1.5, null_max)
    assert res["p_value"] < 0.05
    res_null = st.permutation_test(0.0, null_max)
    assert res_null["p_value"] > 0.05


def test_bootstrap_ci_excludes_zero_for_clear_effect():
    rng = np.random.default_rng(2)
    vals = rng.normal(1.0, 0.2, 200)
    clusters = np.repeat(np.arange(20), 10)
    res = st.clustered_bootstrap(vals, clusters, np.mean, n_boot=500)
    assert res["excludes_zero"]


def test_benjamini_hochberg():
    res = st.benjamini_hochberg([0.001, 0.02, 0.5, 0.9])
    assert res["n_reject"] >= 1


def test_mediation_fraction():
    assert st.mediation_fraction(1.0, 0.3) == pytest.approx(0.7)
    assert st.mediation_fraction(1.0, 1.0) == pytest.approx(0.0)


# ---- cot carriers ----

def test_arithmetic_traces_all_verified():
    rng = np.random.default_rng(0)
    ds = cc.generate_carrier_dataset(60, rng)
    report = cc.verify_dataset(ds)
    assert report["failed"] == 0 and report["verified"] == 60


def test_checksum_and_strict_are_separable():
    rng = np.random.default_rng(1)
    ds = cc.generate_carrier_dataset(5, rng)
    ex = ds[0]
    strict = ex.render(0)
    checksum = cc.render_with_checksum(ex, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert "checksum" not in strict.lower()
    assert "checksum" in checksum.lower()
    # answer unchanged
    assert str(ex.problem.answer) in strict and str(ex.problem.answer) in checksum

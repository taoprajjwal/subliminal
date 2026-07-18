"""Scientific-invariant tests (EXPERIMENT_PLAN.md §10.3).

These fail loudly if the scientific interpretation is invalid. They operate on
synthetic activations / scores so they run without a model, but encode the same
invariants the model-driven pipeline must satisfy.
"""
import numpy as np
import pytest

from subliminal_icl import trait_subspace as tss, patching as pt
from subliminal_icl import statistics as st
from subliminal_icl.semantic_filters import SemanticScanner
from subliminal_icl.token_scoring import target_margin

pytestmark = pytest.mark.scientific


def test_evaluator_specificity_not_all_animals_rise():
    # A target-specific effect must lift the target margin more than a generic
    # shift that raises all animals equally.
    base = {a: -2.0 for a in ["eagle", "cat", "dog", "owl"]}
    specific = dict(base); specific["eagle"] = -0.5
    generic = {a: v + 1.0 for a, v in base.items()}  # everything rises equally
    assert target_margin(specific, "eagle") > target_margin(base, "eagle")
    assert np.isclose(target_margin(generic, "eagle"), target_margin(base, "eagle"))


def test_causal_vector_sign():
    rng = np.random.default_rng(0)
    d = 8
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    P = tss.projector(u[:, None])
    h0 = rng.standard_normal(d)
    hT = h0 + 2.0 * u
    pos = float((pt.patch_in(h0, hT, P, +1.0) - h0) @ u)
    neg = float((pt.patch_in(h0, hT, P, -1.0) - h0) @ u)
    assert pos > 0 > neg  # sign reversal reverses the effect


def test_random_subspace_null_weaker_than_true():
    rng = np.random.default_rng(1)
    d = 40
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    P_true = tss.projector(u[:, None])
    h0 = rng.standard_normal(d)
    hT = h0 + 3.0 * u
    true_eff = abs(float((pt.patch_in(h0, hT, P_true, 1.0) - h0) @ u))
    rand_effs = []
    for _ in range(50):
        P = tss.projector(rng.standard_normal((d, 1)))
        rand_effs.append(abs(float((pt.patch_in(h0, hT, P, 1.0) - h0) @ u)))
    assert true_eff > np.percentile(rand_effs, 95)


def test_search_budget_null_uses_max_not_mean():
    # The selected result must beat the MAX of budget-matched nulls, which is a
    # stricter bar than the mean.
    rng = np.random.default_rng(2)
    null_max = rng.normal(0, 1, size=100)
    observed = float(null_max.max()) + 0.5
    p_vs_max = st.permutation_test(observed, null_max)["p_value"]
    assert p_vs_max < 0.05
    # a value below the null max must not pass
    assert st.permutation_test(float(np.median(null_max)), null_max)["p_value"] > 0.05


def test_semantic_leakage_strict_zero_hits():
    sc = SemanticScanner()
    # a strictly clean numeric context
    assert sc.is_strict_clean("12, 45, 903, 7, 88, 640", ["eagle"])
    # any target word, plural, synonym, or translation must fail strict
    for bad in ["eagle", "eagles", "raptor", "aguila", "an eagle appears"]:
        assert not sc.is_strict_clean(bad, ["eagle"]), bad


def test_shuffled_target_labels_destroy_specificity():
    # gamma should collapse when the diagonal indicator is shuffled.
    rng = np.random.default_rng(3)
    animals = [f"a{i}" for i in range(5)]
    t, j, q, d = [], [], [], []
    for tt in animals:
        for jj in animals:
            for qi in range(6):
                t.append(tt); j.append(jj); q.append(qi)
                d.append((1.0 if tt == jj else 0.0) + rng.normal(0, 0.1))
    real = st.fit_diagonal_model(st.DiagonalObservations(t=t, j=j, q=q, delta=d), n_boot=200)
    # shuffle the measured-animal labels -> diagonal indicator no longer aligns
    j_shuf = list(np.array(j)[rng.permutation(len(j))])
    shuf = st.fit_diagonal_model(st.DiagonalObservations(t=t, j=j_shuf, q=q, delta=d), n_boot=200)
    assert real["gamma"] > 0.5
    assert abs(shuf["gamma"]) < real["gamma"]

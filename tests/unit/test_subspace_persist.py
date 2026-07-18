import numpy as np
import pytest

from subliminal_icl.trait_subspace import TraitSubspace

pytestmark = pytest.mark.unit


def test_trait_subspace_save_load_roundtrip(tmp_path):
    d = 16
    basis = np.linalg.qr(np.random.default_rng(0).standard_normal((d, 2)))[0]
    mu = np.random.default_rng(1).standard_normal(d)
    sub = TraitSubspace("eagle", 12, "final_prefill", "residual", basis=basis,
                        mu_neutral=mu, method="cca", meta={"held_out_score": 0.9})
    prefix = str(tmp_path / "sub")
    sub.save(prefix)
    loaded = TraitSubspace.load(prefix)
    assert loaded.target == "eagle"
    assert loaded.layer == 12
    assert loaded.rank == 2
    assert loaded.method == "cca"
    assert np.allclose(loaded.basis, sub.basis)
    assert np.allclose(loaded.mu_neutral, sub.mu_neutral)
    # scoring behaves identically
    h = np.random.default_rng(2).standard_normal(d)
    assert np.isclose(loaded.signed_score(h), sub.signed_score(h))


def test_trait_subspace_save_load_with_whitening(tmp_path):
    d = 8
    basis = np.linalg.qr(np.random.default_rng(0).standard_normal((d, 1)))[0]
    W = np.eye(d) * 2.0
    sub = TraitSubspace("cat", 6, "final_prefill", "residual", basis=basis,
                        mu_neutral=np.zeros(d), whitening=W)
    prefix = str(tmp_path / "sub")
    sub.save(prefix)
    loaded = TraitSubspace.load(prefix)
    assert loaded.whitening is not None
    assert np.allclose(loaded.whitening, W)

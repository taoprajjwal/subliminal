import numpy as np
import pytest

from subliminal_icl import patching as pt, trait_subspace as tss

pytestmark = pytest.mark.unit


def test_noop_patch_alpha_zero_is_identity():
    rng = np.random.default_rng(0)
    P = tss.projector(rng.standard_normal((8, 2)))
    h0 = rng.standard_normal(8)
    hT = rng.standard_normal(8)
    assert np.allclose(pt.patch_in(h0, hT, P, 0.0), h0)


def test_patch_in_alpha_one_projects_full_shift():
    rng = np.random.default_rng(1)
    d = 6
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    P = tss.projector(u[:, None])
    h0 = rng.standard_normal(d)
    hT = h0 + 3.0 * u  # shift purely along u
    patched = pt.patch_in(h0, hT, P, 1.0)
    # along u we recover the full 3.0 shift
    assert np.isclose(float((patched - h0) @ u), 3.0, atol=1e-8)


def test_ablate_removes_subspace_component():
    rng = np.random.default_rng(2)
    d = 6
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    P = tss.projector(u[:, None])
    h0 = rng.standard_normal(d)
    hT = h0 + 2.5 * u
    ablated = pt.ablate(hT, h0, P)
    assert abs(float((ablated - h0) @ u)) < 1e-8  # component along u removed


def test_random_subspace_effect_smaller_than_true():
    rng = np.random.default_rng(3)
    d = 20
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    P_true = tss.projector(u[:, None])
    P_rand = tss.projector(rng.standard_normal((d, 1)))
    h0 = rng.standard_normal(d)
    hT = h0 + 3.0 * u
    rec_true = abs(float((pt.patch_in(h0, hT, P_true, 1.0) - h0) @ u))
    rec_rand = abs(float((pt.patch_in(h0, hT, P_rand, 1.0) - h0) @ u))
    assert rec_true > rec_rand


def test_dose_response_monotone_along_direction():
    rng = np.random.default_rng(4)
    d = 6
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    P = tss.projector(u[:, None])
    h0 = rng.standard_normal(d)
    hT = h0 + u
    dr = pt.dose_response(h0, hT, P, [0.0, 0.5, 1.0, 2.0])
    vals = [float((dr[a] - h0) @ u) for a in [0.0, 0.5, 1.0, 2.0]]
    assert vals == sorted(vals)

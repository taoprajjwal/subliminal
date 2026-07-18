import numpy as np
import pytest

from subliminal_icl import trait_subspace as tss

pytestmark = pytest.mark.unit


def test_basis_orthonormality():
    rng = np.random.default_rng(0)
    U = tss.orthonormal_basis(rng.standard_normal((10, 3)))
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-8)


def test_projector_idempotence_and_symmetry():
    rng = np.random.default_rng(1)
    P = tss.projector(rng.standard_normal((8, 2)))
    assert np.allclose(P @ P, P, atol=1e-8)
    assert np.allclose(P, P.T, atol=1e-8)


def test_whitening_train_only_and_shape():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 6))
    W, mu = tss.fit_shrinkage_whitening(X, lam=1e-2)
    assert W.shape == (6, 6)
    assert mu.shape == (6,)
    # whitening approximately decorrelates the training data
    Z = tss.whiten(X, W, mu)
    cov = np.cov(Z.T)
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 0.5


def test_sign_alignment_meandiff():
    rng = np.random.default_rng(3)
    d = 5
    u = rng.standard_normal(d)
    pos = u + 0.01 * rng.standard_normal((50, d))
    neg = -u + 0.01 * rng.standard_normal((50, d))
    v = tss.mean_difference_direction(pos, neg)
    assert np.dot(v, u) > 0  # points from neg toward pos
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_random_subspace_orthonormal():
    rng = np.random.default_rng(4)
    U = tss.random_subspace(12, 3, rng)
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-8)


def test_cca_recovers_shared_direction():
    rng = np.random.default_rng(5)
    d, n = 12, 300
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    s = rng.standard_normal(n)
    A = np.outer(s, u) + 0.2 * rng.standard_normal((n, d))
    B = np.outer(s, u) + 0.2 * rng.standard_normal((n, d))
    shared = tss.cca_shared(A, B, rank=1)
    a = shared["A"][:, 0]; a /= np.linalg.norm(a)
    assert abs(np.dot(a, u)) > 0.8
    assert shared["corr"][0] > 0.8


def test_permuted_labels_reduce_heldout_pairing():
    rng = np.random.default_rng(6)
    d, n = 12, 300
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    s = rng.standard_normal(n)
    A = np.outer(s, u) + 0.3 * rng.standard_normal((n, d))
    B = np.outer(s, u) + 0.3 * rng.standard_normal((n, d))
    tr, va = slice(0, 200), slice(200, n)

    def paired_corr(Bmat):
        a = tss.cca_shared(A[tr], Bmat[tr], rank=1)["A"][:, 0]
        return float(np.corrcoef(A[va] @ a, Bmat[va] @ a)[0, 1])

    real = paired_corr(B)
    perm = paired_corr(B[rng.permutation(n)])
    assert real > 0.5
    assert perm < real - 0.2

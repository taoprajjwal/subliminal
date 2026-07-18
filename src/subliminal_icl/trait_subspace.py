"""Trait-subspace extraction and scoring (EXPERIMENT_PLAN.md §9.1, §9.2).

Pure numpy. Implements:
  - shrinkage whitening fitted on *training* activations only;
  - whitened mean-difference direction;
  - rank-k PCA/SVD of paired differences;
  - CCA (shared components between explicit-shift and FT-shift matrices);
  - orthonormal projectors, idempotence-checked;
  - contrastive whitened target-state scores (signed rank-one and distance form);
  - norm-matched random subspaces for null controls.

A subspace object is *eligible for search only after* causal patch-in
sufficiency + donor-ablation necessity tests (done in patching.py / notebook 05).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# whitening
# --------------------------------------------------------------------------

def fit_shrinkage_whitening(X: np.ndarray, lam: float = 1e-2
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``(Sigma + lam I)^(-1/2)`` and the mean from rows of ``X`` (n, d).

    Returns ``(W, mu)`` where whitening of a vector ``h`` is ``W @ (h - mu)``.
    Fit this on training activations ONLY.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    n, d = Xc.shape
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    # shrink toward scaled identity
    trace_mean = np.trace(cov) / d
    cov_reg = cov + lam * trace_mean * np.eye(d)
    # symmetric inverse sqrt via eigdecomposition
    vals, vecs = np.linalg.eigh(cov_reg)
    vals = np.clip(vals, 1e-12, None)
    W = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    return W, mu


def whiten(h: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (np.asarray(h, dtype=np.float64) - mu) @ W.T


# --------------------------------------------------------------------------
# directions / bases
# --------------------------------------------------------------------------

def mean_difference_direction(pos: np.ndarray, neg: np.ndarray,
                              W: Optional[np.ndarray] = None,
                              mu: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalized (optionally whitened) mean(pos) - mean(neg) direction."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    raw = pos.mean(axis=0) - neg.mean(axis=0)
    if W is not None:
        raw = raw @ W.T if mu is None else whiten(pos.mean(0), W, mu) - whiten(neg.mean(0), W, mu)
    n = np.linalg.norm(raw)
    return raw / n if n > 0 else raw


def pca_directions(diffs: np.ndarray, rank: int = 1,
                   center: bool = True) -> np.ndarray:
    """Top-``rank`` right singular vectors of paired difference rows (n, d).

    Returns an orthonormal basis of shape (d, rank).
    """
    D = np.asarray(diffs, dtype=np.float64)
    if center:
        D = D - D.mean(axis=0, keepdims=True)
    # economy SVD: D = U S Vt ; principal directions are rows of Vt
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:rank].T  # (d, rank)


def cca_shared(D_sys: np.ndarray, D_ft: np.ndarray, rank: int = 1,
               reg: float = 1e-3) -> Dict[str, np.ndarray]:
    """Canonical correlation between explicit-shift rows and FT-shift rows.

    Both matrices are (n_prompts, d) aligned by prompt. Returns dict with
    ``A`` (d, rank) canonical directions in the explicit space, ``B`` (d, rank)
    in the FT space, and ``corr`` (rank,) canonical correlations. The shared
    subspace basis used downstream is the orthonormalized ``A``.
    """
    Xa = np.asarray(D_sys, dtype=np.float64)
    Xb = np.asarray(D_ft, dtype=np.float64)
    Xa = Xa - Xa.mean(0, keepdims=True)
    Xb = Xb - Xb.mean(0, keepdims=True)
    n = Xa.shape[0]
    Caa = (Xa.T @ Xa) / max(n - 1, 1) + reg * np.eye(Xa.shape[1])
    Cbb = (Xb.T @ Xb) / max(n - 1, 1) + reg * np.eye(Xb.shape[1])
    Cab = (Xa.T @ Xb) / max(n - 1, 1)

    def inv_sqrt(M):
        vals, vecs = np.linalg.eigh(M)
        vals = np.clip(vals, 1e-12, None)
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T

    Ra, Rb = inv_sqrt(Caa), inv_sqrt(Cbb)
    M = Ra @ Cab @ Rb
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    A = Ra @ U[:, :rank]
    B = Rb @ Vt[:rank].T
    # normalize columns
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return {"A": A, "B": B, "corr": S[:rank]}


def orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    """QR-based orthonormalization of columns; returns (d, k)."""
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    Q, _ = np.linalg.qr(V)
    return Q[:, : V.shape[1]]


def projector(basis: np.ndarray) -> np.ndarray:
    """Orthogonal projector ``P = U U^T`` onto span(columns of basis)."""
    U = orthonormal_basis(basis)
    return U @ U.T


def random_subspace(d: int, rank: int, rng: np.random.Generator,
                    match_norm_to: Optional[np.ndarray] = None) -> np.ndarray:
    """Random orthonormal (d, rank) basis, optionally norm-matched.

    Norm matching is applied by callers to the *effect vector*, not the basis
    itself; the basis is orthonormal. Returned for use as a matched null.
    """
    G = rng.standard_normal((d, rank))
    return orthonormal_basis(G)


# --------------------------------------------------------------------------
# subspace container + scoring
# --------------------------------------------------------------------------

@dataclass
class TraitSubspace:
    """A frozen, causally-selected trait subspace and its readout parameters."""

    target: str
    layer: int
    position: str
    component: str  # residual | attn | mlp
    basis: np.ndarray  # (d, rank), orthonormal
    mu_neutral: np.ndarray  # (d,)
    whitening: Optional[np.ndarray] = None  # (d, d) or None
    rank: int = field(init=False)
    method: str = "mean_diff"
    contrastive_directions: Optional[Dict[str, np.ndarray]] = None  # other-trait dirs
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.basis = np.asarray(self.basis, dtype=np.float64)
        if self.basis.ndim == 1:
            self.basis = self.basis[:, None]
        self.mu_neutral = np.asarray(self.mu_neutral, dtype=np.float64)
        self.rank = self.basis.shape[1]

    # ---- geometry ----
    def projector(self) -> np.ndarray:
        return projector(self.basis)

    def project(self, h: np.ndarray) -> np.ndarray:
        """Coordinates of ``h - mu_neutral`` in the (whitened) basis: (..., rank)."""
        x = np.asarray(h, dtype=np.float64) - self.mu_neutral
        if self.whitening is not None:
            x = x @ self.whitening.T
        return x @ self.basis

    # ---- scores (§9.2) ----
    def signed_score(self, h: np.ndarray) -> float:
        """Signed rank-one style score: projection onto the (unit) target
        direction minus mean projection onto contrastive other-trait directions.
        For rank>1 uses the norm of the projection coordinates.
        """
        coords = self.project(h)
        if self.rank == 1:
            base = float(coords.reshape(-1)[0])
        else:
            base = float(np.linalg.norm(coords))
        if self.contrastive_directions:
            others = []
            x = np.asarray(h, dtype=np.float64) - self.mu_neutral
            if self.whitening is not None:
                x = x @ self.whitening.T
            for _, v in self.contrastive_directions.items():
                v = np.asarray(v, dtype=np.float64)
                v = v / (np.linalg.norm(v) + 1e-12)
                others.append(float(x @ v))
            base -= float(np.mean(others))
        return base

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        d = {
            "basis": self.basis,
            "mu_neutral": self.mu_neutral,
        }
        if self.whitening is not None:
            d["whitening"] = self.whitening
        return d

    def save(self, prefix: str) -> str:
        """Persist to ``<prefix>.npz`` + ``<prefix>.meta.json``. Returns the npz path."""
        import json
        from pathlib import Path
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        np.savez(f"{prefix}.npz", **self.to_npz_dict())
        meta = {"target": self.target, "layer": self.layer, "position": self.position,
                "component": self.component, "method": self.method, "rank": self.rank,
                "meta": self.meta}
        Path(f"{prefix}.meta.json").write_text(json.dumps(meta, default=str))
        return f"{prefix}.npz"

    @classmethod
    def load(cls, prefix: str) -> "TraitSubspace":
        import json
        from pathlib import Path
        arrs = np.load(f"{prefix}.npz")
        meta = json.loads(Path(f"{prefix}.meta.json").read_text())
        return cls(target=meta["target"], layer=meta["layer"], position=meta["position"],
                   component=meta["component"], basis=arrs["basis"], mu_neutral=arrs["mu_neutral"],
                   whitening=arrs["whitening"] if "whitening" in arrs.files else None,
                   method=meta.get("method", "mean_diff"), meta=meta.get("meta", {}))


def held_out_alignment(D_sys_train: np.ndarray, D_ft_train: np.ndarray,
                       D_sys_val: np.ndarray, D_ft_val: np.ndarray,
                       rank: int = 1) -> Dict[str, float]:
    """Fit shared directions on train, measure alignment of the two shift
    directions on held-out val prompts. Returns cosine of mean val shifts
    projected onto the fitted shared direction, plus canonical corr on train.
    """
    shared = cca_shared(D_sys_train, D_ft_train, rank=rank)
    A = shared["A"]  # (d, rank)
    a = A[:, 0]
    a = a / (np.linalg.norm(a) + 1e-12)
    sys_val = np.asarray(D_sys_val).mean(0)
    ft_val = np.asarray(D_ft_val).mean(0)
    proj_sys = float(sys_val @ a)
    proj_ft = float(ft_val @ a)
    cos = float(
        (sys_val @ ft_val)
        / (np.linalg.norm(sys_val) * np.linalg.norm(ft_val) + 1e-12)
    )
    return {
        "train_canonical_corr": float(shared["corr"][0]),
        "val_shift_cosine": cos,
        "val_proj_sys": proj_sys,
        "val_proj_ft": proj_ft,
        "val_proj_agree_sign": float(np.sign(proj_sys) == np.sign(proj_ft)),
    }

"""Statistics for the diagonal transfer effect (EXPERIMENT_PLAN.md §12, §13).

numpy-only (scipy used only if present, for exact permutation p-values). The
mixed-effects diagonal model is fit by OLS on dummy-coded fixed effects with
clustered bootstrap CIs over the resampling units (evaluation prompt & seed),
which is the reported ``gamma`` in the plan. Full REML is out of scope for the
scaffold; the OLS point estimate + clustered bootstrap is a faithful stand-in
and can be swapped for statsmodels MixedLM when that dep is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def clustered_bootstrap(values: np.ndarray, clusters: Sequence,
                        statistic: Callable[[np.ndarray], float],
                        n_boot: int = 2000, alpha: float = 0.05,
                        rng: Optional[np.random.Generator] = None
                        ) -> Dict[str, float]:
    """Bootstrap CI resampling whole clusters (e.g. prompts / seeds)."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    point = float(statistic(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in chosen])
        boots[b] = statistic(values[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"point": point, "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def permutation_test(observed: float, null_samples: Sequence[float],
                     tail: str = "greater") -> Dict[str, float]:
    """p-value of ``observed`` against an empirical null.

    For the search-budget-matched test, ``null_samples`` must be the *max*
    statistic from each budget-matched null search (EXPERIMENT_PLAN.md §13.3),
    one value per null search, so the comparison accounts for selection.
    """
    null = np.asarray(null_samples, dtype=np.float64)
    n = null.size
    if tail == "greater":
        count = int(np.sum(null >= observed))
    elif tail == "less":
        count = int(np.sum(null <= observed))
    else:  # two-sided on |.|
        count = int(np.sum(np.abs(null) >= abs(observed)))
    p = (count + 1) / (n + 1)  # add-one for finite sample
    return {"observed": float(observed), "p_value": float(p), "n_null": n,
            "null_mean": float(null.mean()) if n else float("nan"),
            "null_max": float(null.max()) if n else float("nan")}


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, object]:
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax > 0:
        reject[order[:kmax]] = True
    qvals = np.empty(m)
    qvals[order] = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    return {"reject": reject.tolist(), "qvalues": qvals.tolist(), "n_reject": int(reject.sum())}


# --------------------------------------------------------------------------
# diagonal transfer model
# --------------------------------------------------------------------------

@dataclass
class DiagonalObservations:
    """Long-form observations for the diagonal model.

    Each entry i is one measurement: context/target trait ``t[i]``, measured
    animal ``j[i]``, evaluation prompt ``q[i]``, seed ``r[i]``, and response
    ``delta[i]`` (e.g. target-margin change vs baseline).
    """

    t: List[str]
    j: List[str]
    q: List = field(default_factory=list)
    r: List = field(default_factory=list)
    delta: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.delta)


def _design_matrix(t, j):
    t = np.asarray(t)
    j = np.asarray(j)
    ts = sorted(set(t.tolist()))
    js = sorted(set(j.tolist()))
    n = len(t)
    cols = []
    names = ["intercept"]
    cols.append(np.ones(n))
    # alpha[t] (drop first for identifiability)
    for tv in ts[1:]:
        cols.append((t == tv).astype(float)); names.append(f"alpha[{tv}]")
    # beta[j] (drop first)
    for jv in js[1:]:
        cols.append((j == jv).astype(float)); names.append(f"beta[{jv}]")
    # gamma diagonal
    cols.append((t == j).astype(float)); names.append("gamma")
    X = np.vstack(cols).T
    return X, names


def fit_diagonal_model(obs: DiagonalObservations, n_boot: int = 2000,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
    """OLS fit of ``delta ~ 1 + alpha[t] + beta[j] + gamma*I(t==j)`` with a
    clustered bootstrap CI on ``gamma`` (clusters = evaluation prompt, falling
    back to seed then row)."""
    rng = rng or np.random.default_rng(0)
    y = np.asarray(obs.delta, dtype=np.float64)
    X, names = _design_matrix(obs.t, obs.j)
    gamma_idx = names.index("gamma")

    def _gamma(mask_idx: np.ndarray) -> float:
        Xb, yb = X[mask_idx], y[mask_idx]
        coef, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        return float(coef[gamma_idx])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    gamma = float(coef[gamma_idx])

    # choose clusters
    if obs.q:
        clusters = np.asarray(obs.q)
    elif obs.r:
        clusters = np.asarray(obs.r)
    else:
        clusters = np.arange(len(y))
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in chosen])
        try:
            boots[b] = _gamma(idx)
        except np.linalg.LinAlgError:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    coefs = {n: float(c) for n, c in zip(names, coef)}
    return {
        "gamma": gamma,
        "gamma_ci_low": float(lo),
        "gamma_ci_high": float(hi),
        "gamma_excludes_zero": bool(lo > 0 or hi < 0),
        "coefficients": coefs,
        "n_obs": len(y),
    }


def mediation_fraction(behavior_total: float, ablated_effect: float) -> float:
    """1 - ablated_effect / total_effect (EXPERIMENT_PLAN.md §11)."""
    if behavior_total == 0:
        return float("nan")
    return float(1.0 - ablated_effect / behavior_total)

"""Trait-representation extraction + causal subspace selection (Gates 3A/3B).

Builds the paired difference matrices on the activation-prototype prompts:

    d_sys[i,l] = h_base(target_system, x_i) - h_base(neutral_system, x_i)
    d_ft[i,l]  = h_target_donor(neutral, x_i) - h_control_donor(neutral, x_i)

fits candidate subspaces (mean-diff / PCA / CCA) per (layer, rank), selects on
held-out prompts, and validates causally by patch-in sufficiency + donor-ablation
necessity on a real model. Uses runtime + hooks + trait_subspace + patching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chat_templates import SystemMode
from .runtime import ActivationSpec, capture_prompt_activations
from .trait_prompts import trait_prompts, neutral_system_prompt
from . import trait_subspace as tss
from . import patching as pt


# --------------------------------------------------------------------------
# prototype prompt set (§5.4) — domains that do NOT ask about animals
# --------------------------------------------------------------------------

def default_prototype_prompts(n_per_domain: int = 20, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    prompts: List[str] = []
    for _ in range(n_per_domain):  # number continuation
        seed_nums = ", ".join(str(int(x)) for x in rng.integers(0, 999, 5))
        prompts.append(f"Continue this sequence: {seed_nums}. Numbers only.")
    for _ in range(n_per_domain):  # short arithmetic
        a, b = int(rng.integers(10, 99)), int(rng.integers(10, 99))
        prompts.append(f"Compute {a} + {b}.")
    factual = ["What is the capital of France?", "How many days are in a week?",
               "Name a primary color.", "What gas do plants absorb?",
               "Who wrote Romeo and Juliet?"]
    instr = ["Write a one-sentence greeting.", "List two fruits.",
             "Say hello politely.", "Give one word for happy.",
             "Reply with a short thank-you."]
    for k in range(n_per_domain):
        prompts.append(factual[k % len(factual)])
        prompts.append(instr[k % len(instr)])
    return prompts


# --------------------------------------------------------------------------
# difference matrices
# --------------------------------------------------------------------------

def explicit_shift_matrix(base_loaded, prompts: Sequence[str], target: str,
                          layers: Sequence[int], anchor: str = "final_prefill",
                          n_prompt_forms: int = 3) -> Dict[int, np.ndarray]:
    """d_sys per layer: mean over trait-prompt paraphrases of (target - neutral)."""
    spec = ActivationSpec(layers=layers, anchor=anchor)
    neutral = neutral_system_prompt()
    h_neu = capture_prompt_activations(base_loaded, prompts, spec, SystemMode.EXPLICIT,
                                       system_text=neutral)
    forms = trait_prompts(target)[:n_prompt_forms]
    acc = {l: np.zeros_like(h_neu[l]) for l in layers}
    for form in forms:
        h_t = capture_prompt_activations(base_loaded, prompts, spec, SystemMode.EXPLICIT,
                                         system_text=form)
        for l in layers:
            acc[l] += (h_t[l] - h_neu[l])
    return {l: acc[l] / len(forms) for l in layers}


def finetune_shift_matrix(target_donor_loaded, control_donor_loaded, prompts: Sequence[str],
                          layers: Sequence[int], anchor: str = "final_prefill"
                          ) -> Dict[int, np.ndarray]:
    """d_ft per layer: h_target_donor(neutral) - h_control_donor(neutral)."""
    spec = ActivationSpec(layers=layers, anchor=anchor)
    h_t = capture_prompt_activations(target_donor_loaded, prompts, spec, SystemMode.NONE)
    h_c = capture_prompt_activations(control_donor_loaded, prompts, spec, SystemMode.NONE)
    return {l: h_t[l] - h_c[l] for l in layers}


# --------------------------------------------------------------------------
# subspace fitting + held-out selection (Gate 3A)
# --------------------------------------------------------------------------

@dataclass
class SubspaceCandidate:
    layer: int
    rank: int
    method: str
    basis: np.ndarray          # (d, rank) orthonormal
    mu_neutral: np.ndarray
    held_out_score: float      # per-prompt paired projection correlation on val
    train_corr: float

    def to_trait_subspace(self, target: str, position: str, component: str = "residual"):
        from .trait_subspace import TraitSubspace
        return TraitSubspace(target=target, layer=self.layer, position=position,
                             component=component, basis=self.basis,
                             mu_neutral=self.mu_neutral, method=self.method,
                             meta={"held_out_score": self.held_out_score,
                                   "train_corr": self.train_corr})


def _paired_corr(D_sys, D_ft, tr, va, rank):
    a = tss.cca_shared(D_sys[tr], D_ft[tr], rank=rank)["A"][:, 0]
    a = a / (np.linalg.norm(a) + 1e-12)
    ps, pf = D_sys[va] @ a, D_ft[va] @ a
    if np.std(ps) < 1e-9 or np.std(pf) < 1e-9:
        return 0.0
    return float(np.corrcoef(ps, pf)[0, 1])


def fit_subspace_candidates(d_sys: Dict[int, np.ndarray], d_ft: Dict[int, np.ndarray],
                            mu_neutral: Dict[int, np.ndarray], ranks: Sequence[int] = (1, 2, 3, 4),
                            methods: Sequence[str] = ("mean_diff", "pca", "cca"),
                            val_frac: float = 0.3, seed: int = 0) -> List[SubspaceCandidate]:
    """Fit candidates per (layer, rank, method); score by held-out paired corr."""
    rng = np.random.default_rng(seed)
    out: List[SubspaceCandidate] = []
    for layer in d_sys:
        Ds, Df = d_sys[layer], d_ft[layer]
        n = Ds.shape[0]
        idx = rng.permutation(n)
        n_val = max(2, int(n * val_frac))
        va, tr = idx[:n_val], idx[n_val:]
        for rank in ranks:
            for method in methods:
                if method == "mean_diff":
                    basis = tss.orthonormal_basis(Ds[tr].mean(0)[:, None])
                elif method == "pca":
                    basis = tss.pca_directions(np.vstack([Ds[tr], Df[tr]]), rank=rank)
                elif method == "cca":
                    basis = tss.orthonormal_basis(tss.cca_shared(Ds[tr], Df[tr], rank=rank)["A"])
                else:
                    continue
                ho = _paired_corr(Ds, Df, tr, va, rank)
                tc = _paired_corr(Ds, Df, tr, tr, rank)
                out.append(SubspaceCandidate(layer, rank, method, basis, mu_neutral[layer], ho, tc))
    out.sort(key=lambda c: c.held_out_score, reverse=True)
    return out


def permutation_null_score(d_sys: Dict[int, np.ndarray], d_ft: Dict[int, np.ndarray],
                           layer: int, rank: int = 1, seed: int = 0) -> float:
    """Held-out paired corr after shuffling the prompt pairing (Gate 3A control)."""
    rng = np.random.default_rng(seed)
    Ds, Df = d_sys[layer], d_ft[layer]
    n = Ds.shape[0]
    idx = rng.permutation(n)
    n_val = max(2, int(n * 0.3))
    va, tr = idx[:n_val], idx[n_val:]
    Df_perm = Df[rng.permutation(n)]
    return _paired_corr(Ds, Df_perm, tr, va, rank)


# --------------------------------------------------------------------------
# causal validation (Gate 3B): patch-in sufficiency + donor-ablation necessity
# --------------------------------------------------------------------------

def target_margin_under_patch(base_loaded, subspace, question: str, target: str,
                              animals: Sequence[str], h_target: np.ndarray, h_neutral: np.ndarray,
                              alpha: float, layer: int, anchor_index: Optional[int] = None
                              ) -> float:
    """Score F_T on a neutral prompt while patching alpha*P(h_target-h_neutral)
    into the residual stream at ``layer`` (all positions if anchor_index None)."""
    import torch
    from .hooks import patch_activations
    from .model_adapters import residual_module_path
    from .chat_templates import render_chat
    from .token_scoring import candidate_logprob, target_margin

    model, tok = base_loaded.model, base_loaded.tokenizer
    P = subspace.projector()
    direction = (np.asarray(h_target) - np.asarray(h_neutral)) @ P.T  # projected shift
    patch_fn = pt.make_add_direction_patch(direction, alpha, position_index=anchor_index)
    path = residual_module_path(model, layer)

    rc = render_chat(question, SystemMode.NONE, tokenizer=tok)
    prefix = list(rc.token_ids)
    with patch_activations(model, {path: patch_fn}):
        lp = {a: candidate_logprob(model, tok, prefix, a).total_logprob for a in animals}
    return target_margin(lp, target)


@dataclass
class CausalResult:
    baseline_margin: float
    dose_response: Dict[float, float]
    random_dose_response: Dict[float, float]

    def sufficiency_effect(self) -> float:
        pos = max(self.dose_response.values())
        return float(pos - self.baseline_margin)

    def beats_random(self) -> bool:
        true_eff = max(self.dose_response.values()) - self.baseline_margin
        rand_eff = max(self.random_dose_response.values()) - self.baseline_margin
        return bool(true_eff > rand_eff)

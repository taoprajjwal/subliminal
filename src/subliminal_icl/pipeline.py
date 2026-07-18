"""End-to-end pilot pipeline (EXPERIMENT_PLAN.md §18, Gates 2-7).

Each phase runs on a live model, produces artifacts under a run directory, and
writes a real gate checkpoint with model evidence. Phases stop the pipeline on a
non-PASS result (the spec's "stop at each gate" discipline).

Callable at any scale: the same code runs on Qwen2.5-0.5B (smoke, seconds) and
Qwen2.5-7B (pilot, hours) — only the config sizes differ. run_pilot.py drives it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import json
import numpy as np

from . import ANIMALS
from .config import Config
from .gates import run_gate_checks, PASS
from .manifests import Manifest
from .seeds import seed_everything, SeedBundle


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def number_prompts(n: int, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    tpls = [
        "These numbers follow a sequence: {s}. Add {k} more values (0-999). Numbers only, comma-separated.",
        "Continue this pattern: {s}. Give {k} more integers in [0,999], comma-separated, numbers only.",
        "Sequence so far: {s}. Output the next {k} values (0-999) as a comma-separated list. No words.",
    ]
    out = []
    for i in range(n):
        k = int(rng.integers(6, 11))
        seeds = ", ".join(str(int(x)) for x in rng.integers(0, 999, int(rng.integers(4, 9))))
        out.append(tpls[i % len(tpls)].format(s=seeds, k=k))
    return out


def eval_dev_questions(config: Config) -> List[dict]:
    from .trait_prompts import eval_questions, EVAL_FAMILIES
    per = 3 if config.modes.fast_dev_run else config.get("data", {})
    animals = list(ANIMALS)
    out = []
    for fam in EVAL_FAMILIES:
        qs = eval_questions(fam, animals)
        for q in qs[:3]:
            out.append({"prompt": q, "family": fam})
    return out


@dataclass
class PipelineContext:
    config: Config
    model_id: str
    run_dir: Path
    target: str = "eagle"
    fast: bool = True
    base_loaded: object = None
    subspace: object = None
    artifacts: Dict[str, object] = field(default_factory=dict)

    @property
    def evidence(self) -> Dict[str, str]:
        return {"model_id": self.model_id, "run_dir": str(self.run_dir)}

    def load_base(self):
        if self.base_loaded is None:
            from .model_adapters import load_model
            attn = self.config.get("model.attn_implementation", "eager")
            self.base_loaded = load_model(self.model_id, attn_implementation=attn)
        return self.base_loaded

    # --- cross-process resumability: persist json-able artifacts + subspace ---
    def persist(self):
        state = {k: v for k, v in self.artifacts.items() if _jsonable(v)}
        (self.run_dir / "run_state.json").write_text(json.dumps(state, default=str, indent=2))
        if self.subspace is not None:
            self.subspace.save(str(self.run_dir / "subspace_v1"))

    def restore(self):
        sp = self.run_dir / "run_state.json"
        if sp.exists():
            self.artifacts.update(json.loads(sp.read_text()))
        subp = self.run_dir / "subspace_v1.meta.json"
        if subp.exists() and self.subspace is None:
            from .trait_subspace import TraitSubspace
            self.subspace = TraitSubspace.load(str(self.run_dir / "subspace_v1"))
        return self


def _jsonable(v) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------

def phase_02_donor(ctx: PipelineContext):
    """Gate 2: train target + neutral-control donors, check target-specific transfer."""
    from . import donor as D
    from .model_adapters import load_model

    lm = ctx.load_base()
    n = 16 if ctx.fast else ctx.config.get("scoring.pool_size_initial", 2000)
    prompts = number_prompts(min(n, 64) if ctx.fast else n, seed=1)
    from .trait_prompts import trait_prompts, neutral_system_prompt

    tgt_rows = D.build_teacher_dataset(lm, prompts, trait_prompts(ctx.target)[0], ctx.target, seed=1)
    neu_rows = D.build_teacher_dataset(lm, prompts, neutral_system_prompt(), ctx.target, seed=2)
    if len(tgt_rows) < 4 or len(neu_rows) < 4:
        gs = run_gate_checks("gate_02_donor_transfer", "Donor target-specific transfer",
            [("enough_teacher_rows", False, f"target={len(tgt_rows)} neutral={len(neu_rows)} (<4)")],
            config={"model": ctx.model_id}, write=True, evidence=ctx.evidence)
        return gs

    lcfg = ctx.config.get("donor.lora", {})
    cfg = D.LoRAConfig(rank=lcfg.get("rank", 8 if ctx.fast else 16),
                       alpha=lcfg.get("alpha", 32), lr=lcfg.get("lr", 1e-3 if ctx.fast else 1e-4),
                       epochs=1 if ctx.fast else lcfg.get("epochs", 1),
                       batch_size=2 if ctx.fast else 4)
    tdir = str(ctx.run_dir / "donor_target"); cdir = str(ctx.run_dir / "donor_control")
    mt = D.train_lora_donor(ctx.model_id, tgt_rows, cfg, tdir)
    mc = D.train_lora_donor(ctx.model_id, neu_rows, cfg, cdir)
    ctx.artifacts["donor_target_dir"] = tdir
    ctx.artifacts["donor_control_dir"] = cdir

    # transfer eval: target donor should lift F_target vs base; control should not
    donor_t = D.load_donor(ctx.model_id, tdir)
    evalq = eval_dev_questions(ctx.config)
    animals = list(ANIMALS)
    t_transfer = D.evaluate_donor_transfer(donor_t, lm, evalq, ctx.target, animals)
    del donor_t
    donor_c = D.load_donor(ctx.model_id, cdir)
    c_transfer = D.evaluate_donor_transfer(donor_c, lm, evalq, ctx.target, animals)
    del donor_c
    ctx.artifacts["donor_transfer"] = {"target": t_transfer, "control": c_transfer}

    target_specific = t_transfer["target_lift"] > 0 and t_transfer["target_lift"] > c_transfer["target_lift"]
    gs = run_gate_checks("gate_02_donor_transfer", "Donor target-specific transfer", [
        ("target_donor_lifts", t_transfer["target_lift"] > 0, f"lift={t_transfer['target_lift']:.3f}"),
        ("target_specific_vs_control", target_specific,
         f"target={t_transfer['target_lift']:.3f} control={c_transfer['target_lift']:.3f}"),
    ], config={"model": ctx.model_id, "n_target_rows": len(tgt_rows)}, write=True, evidence=ctx.evidence)
    return gs


def phase_03a_extract(ctx: PipelineContext):
    """Gate 3A: build d_sys/d_ft, fit subspace, held-out vs permutation-null."""
    from . import extraction as X, donor as D
    from .runtime import ActivationSpec, capture_prompt_activations
    from .chat_templates import SystemMode
    from .trait_prompts import neutral_system_prompt

    lm = ctx.load_base()
    layers = list(ctx.config.get("subspace.layers_to_search", [ (lm.num_layers*2)//3 ]))
    layers = [l for l in layers if l < lm.num_layers] or [lm.num_layers // 2]
    n_proto = 6 if ctx.fast else 100
    prompts = X.default_prototype_prompts(n_per_domain=n_proto, seed=0)

    d_sys = X.explicit_shift_matrix(lm, prompts, ctx.target, layers, n_prompt_forms=3)

    # d_ft requires donors from phase 2
    tdir = ctx.artifacts.get("donor_target_dir"); cdir = ctx.artifacts.get("donor_control_dir")
    if not (tdir and cdir):
        gs = run_gate_checks("gate_03a_shared_subspace", "Shared explicit/FT subspace",
            [("donors_available", False, "phase_02 donors missing")],
            config={"model": ctx.model_id}, write=True, evidence=ctx.evidence)
        return gs
    donor_t = D.load_donor(ctx.model_id, tdir); donor_c = D.load_donor(ctx.model_id, cdir)
    d_ft = X.finetune_shift_matrix(donor_t, donor_c, prompts, layers)
    del donor_t, donor_c

    spec = ActivationSpec(layers=layers, anchor="final_prefill")
    mu = {l: capture_prompt_activations(lm, prompts, ActivationSpec(layers=[l]), SystemMode.EXPLICIT,
                                        system_text=neutral_system_prompt())[l].mean(0) for l in layers}
    ranks = list(ctx.config.get("subspace.ranks", [1, 2, 3, 4]))
    cands = X.fit_subspace_candidates(d_sys, d_ft, mu, ranks=ranks)
    best = cands[0]
    null = X.permutation_null_score(d_sys, d_ft, best.layer, best.rank)
    ctx.subspace = best.to_trait_subspace(ctx.target, position="final_prefill")
    # persist subspace (npz + meta.json so a later job can restore it)
    ctx.subspace.save(str(ctx.run_dir / "subspace_v1"))
    ctx.artifacts["subspace"] = {"layer": best.layer, "rank": best.rank, "method": best.method,
                                 "held_out": best.held_out_score, "perm_null": null}

    gs = run_gate_checks("gate_03a_shared_subspace", "Shared explicit/FT subspace", [
        ("held_out_alignment", best.held_out_score > 0.3, f"corr={best.held_out_score:.3f} L{best.layer} r{best.rank} {best.method}"),
        ("beats_permutation_null", best.held_out_score > null + 0.15, f"real={best.held_out_score:.3f} null={null:.3f}"),
    ], config={"model": ctx.model_id, "layers": layers}, write=True, evidence=ctx.evidence)
    return gs


def phase_03b_causal(ctx: PipelineContext):
    """Gate 3B: patch-in sufficiency + random-subspace control on held-out queries."""
    from . import extraction as X, trait_subspace as tss
    from .runtime import ActivationSpec, capture_prompt_activations
    from .chat_templates import SystemMode
    from .trait_prompts import trait_prompts, neutral_system_prompt

    lm = ctx.load_base()
    if ctx.subspace is None:
        gs = run_gate_checks("gate_03b_causal_subspace", "Causal sufficiency + necessity",
            [("subspace_available", False, "phase_03a subspace missing")],
            config={"model": ctx.model_id}, write=True, evidence=ctx.evidence)
        return gs
    L = ctx.subspace.layer
    prompts = X.default_prototype_prompts(n_per_domain=4, seed=3)
    h_t = capture_prompt_activations(lm, prompts, ActivationSpec(layers=[L]), SystemMode.EXPLICIT,
                                     system_text=trait_prompts(ctx.target)[0])[L].mean(0)
    h_n = capture_prompt_activations(lm, prompts, ActivationSpec(layers=[L]), SystemMode.EXPLICIT,
                                     system_text=neutral_system_prompt())[L].mean(0)
    animals = list(ANIMALS)
    questions = [e["prompt"] for e in eval_dev_questions(ctx.config)][:4]
    alphas = ctx.config.get("causal_validation.alphas", [0.0, 2.0, 4.0, 6.0])
    alphas = [a for a in alphas if a >= 0]

    def mean_margin(sub, alpha):
        return float(np.mean([X.target_margin_under_patch(lm, sub, q, ctx.target, animals,
                                                          h_t, h_n, alpha, L) for q in questions]))
    base = mean_margin(ctx.subspace, 0.0)
    dr = {float(a): mean_margin(ctx.subspace, a) for a in alphas}
    rng = np.random.default_rng(0)
    sub_rand = tss.TraitSubspace(ctx.target, L, "final_prefill", "residual",
                                 basis=tss.random_subspace(lm.hidden_size, ctx.subspace.rank, rng),
                                 mu_neutral=ctx.subspace.mu_neutral)
    dr_rand = {float(a): mean_margin(sub_rand, a) for a in alphas}
    suff = max(dr.values()) - base
    rand_eff = max(dr_rand.values()) - base
    min_eff = ctx.config.get("causal_validation.sufficiency_min_effect_nats", 0.2)
    ctx.artifacts["causal"] = {"baseline": base, "dose_response": dr, "random_dose_response": dr_rand}

    gs = run_gate_checks("gate_03b_causal_subspace", "Causal sufficiency + partial necessity", [
        ("sufficiency", suff >= min_eff, f"max lift={suff:.3f} (>= {min_eff})"),
        ("beats_random_subspace", suff > rand_eff, f"true={suff:.3f} random={rand_eff:.3f}"),
    ], config={"model": ctx.model_id, "layer": L, "alphas": alphas}, write=True, evidence=ctx.evidence)
    return gs


def phase_04b_score(ctx: PipelineContext):
    """Gate 4B: score candidate rows; high-score rows reconstruct state > low-score."""
    from . import donor as D
    from .search_runtime import score_candidate_rows
    from .trait_prompts import trait_prompts, neutral_system_prompt

    lm = ctx.load_base()
    if ctx.subspace is None:
        gs = run_gate_checks("gate_04b_candidate_variance", "Candidate score variance",
            [("subspace_available", False, "no subspace")], config={"model": ctx.model_id},
            write=True, evidence=ctx.evidence)
        return gs
    # candidate pool: target-teacher + neutral-teacher rows (numbers only)
    n = 12 if ctx.fast else ctx.config.get("scoring.pool_size_initial", 2000)
    prompts = number_prompts(min(n, 40) if ctx.fast else n, seed=7)
    tgt_rows = D.build_teacher_dataset(lm, prompts, trait_prompts(ctx.target)[0], ctx.target, seed=7)
    neu_rows = D.build_teacher_dataset(lm, prompts, neutral_system_prompt(), ctx.target, seed=8)
    rows = ([{"row_id": f"t{i}", "assistant_text": r.assistant_text, "source_label": "target"} for i, r in enumerate(tgt_rows)] +
            [{"row_id": f"n{i}", "assistant_text": r.assistant_text, "source_label": "neutral"} for i, r in enumerate(neu_rows)])
    diag = ctx.config.get("scoring.diagnostic_queries", ["Please restate the last list of numbers.", "What is 2 + 2?"])
    scores = score_candidate_rows(lm, ctx.subspace, rows, ctx.target, diag)
    ctx.artifacts["candidate_scores"] = [s.to_dict() for s in scores]
    (ctx.run_dir / "candidate_scores.json").write_text(json.dumps(ctx.artifacts["candidate_scores"], indent=2))

    finite = [s for s in scores if np.isfinite(s.total)]
    tot = sorted(finite, key=lambda s: s.total)
    hi = np.mean([s.state_score for s in tot[-max(1, len(tot)//3):]])
    lo = np.mean([s.state_score for s in tot[:max(1, len(tot)//3)]])
    gs = run_gate_checks("gate_04b_candidate_variance", "Candidate score variance + specificity", [
        ("has_variance", float(np.std([s.total for s in finite])) > 1e-6, "nonzero score spread"),
        ("high_reconstructs_more", hi > lo, f"high state={hi:.3f} low state={lo:.3f}"),
    ], config={"model": ctx.model_id, "n_rows": len(rows)}, write=True, evidence=ctx.evidence)
    ctx.artifacts["candidate_rows"] = rows
    return gs


def phase_05_search(ctx: PipelineContext):
    """Gate 5: beam-search a selection-only context; beat matched random null."""
    from . import beam_search as bs
    from .search_runtime import make_context_score_fn

    lm = ctx.load_base()
    rows = ctx.artifacts.get("candidate_rows")
    if not rows or ctx.subspace is None:
        gs = run_gate_checks("gate_05_selection_reconstruction", "Selection beats null",
            [("inputs_available", False, "no candidate rows / subspace")],
            config={"model": ctx.model_id}, write=True, evidence=ctx.evidence)
        return gs
    rows_by_id = {r["row_id"]: r for r in rows}
    for i, r in enumerate(rows):
        r.setdefault("source_index", i)  # distinct sources so search can grow
    source_of = {r["row_id"]: str(r["source_index"]) for r in rows}
    diag = ctx.config.get("scoring.diagnostic_queries", ["Please restate the last list of numbers.", "What is 2 + 2?"])
    score_fn = make_context_score_fn(lm, ctx.subspace, rows_by_id, diag)
    propose = lambda beam: list(rows_by_id)
    K = 2 if ctx.fast else 8
    cfg = bs.SearchConfig(beam_width=4 if ctx.fast else 16, proposal_count=len(rows_by_id), max_steps=K)
    cons = bs.DiversityConstraints(max_from_one_source=4)
    st = bs.beam_search(score_fn, propose, cfg, constraints=cons, source_of=source_of)
    best = bs.best_beam(st)

    # matched null: same budget, shuffled-target subspace (random basis)
    from . import trait_subspace as tss
    rng = np.random.default_rng(0)
    sub_null = tss.TraitSubspace(ctx.target, ctx.subspace.layer, "final_prefill", "residual",
                                 basis=tss.random_subspace(lm.hidden_size, ctx.subspace.rank, rng),
                                 mu_neutral=ctx.subspace.mu_neutral)
    null_fn = make_context_score_fn(lm, sub_null, rows_by_id, diag)
    st_null = bs.beam_search(null_fn, propose, cfg, constraints=cons, source_of=source_of)
    # score the null-selected context under the REAL subspace for a fair comparison
    null_ctx_score = score_fn(bs.best_beam(st_null).items)

    context_text = "\n".join(rows_by_id[i]["assistant_text"] for i in best.items)
    (ctx.run_dir / "selected_context.txt").write_text(context_text)
    ctx.artifacts["selected_context"] = context_text
    ctx.artifacts["search"] = {"best_items": best.items, "best_score": best.score,
                               "null_context_score": null_ctx_score}
    gs = run_gate_checks("gate_05_selection_reconstruction", "Selection beats matched null", [
        ("beats_null", best.score > null_ctx_score, f"selected={best.score:.3f} null={null_ctx_score:.3f}"),
        ("strict_leakage_free", _leak_free(context_text, ctx.target), "no strict leakage in context"),
    ], config={"model": ctx.model_id, "K": K}, write=True, evidence=ctx.evidence)
    return gs


def _leak_free(text, target):
    from .semantic_filters import SemanticScanner
    return SemanticScanner().is_strict_clean(text, targets=[target])


def phase_07_clean_replay(ctx: PipelineContext):
    """Gates 6/7: clean text-only replay in a fresh subprocess (import-guarded)."""
    import subprocess, sys
    ctxt = ctx.artifacts.get("selected_context")
    if ctxt is None and (ctx.run_dir / "selected_context.txt").exists():
        ctxt = (ctx.run_dir / "selected_context.txt").read_text()
        ctx.artifacts["selected_context"] = ctxt
    if ctxt is None:
        gs = run_gate_checks("gate_07_clean_replay_behavior", "Clean text-only replay",
            [("context_available", False, "no selected context")],
            config={"model": ctx.model_id}, write=True, evidence=ctx.evidence)
        return gs
    ctx_path = ctx.run_dir / "selected_context.txt"
    eval_path = Path("data/splits/animal_preference_eval_dev.jsonl")
    if not eval_path.exists():
        from .trait_prompts import eval_questions, EVAL_FAMILIES
        eval_path = ctx.run_dir / "eval_dev.jsonl"
        with open(eval_path, "w") as f:
            for fam in EVAL_FAMILIES:
                for q in eval_questions(fam, list(ANIMALS))[:3]:
                    f.write(json.dumps({"prompt": q, "family": fam}) + "\n")
    out = ctx.run_dir / "clean_replay"
    r = subprocess.run([sys.executable, "scripts/clean_replay_eval.py",
                        "--model", ctx.model_id, "--context", str(ctx_path),
                        "--eval", str(eval_path), "--target", ctx.target, "--output", str(out)],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    detail = "clean replay ran" if ok else r.stderr.strip()[-200:]
    # read diagonal if produced
    diag = None
    res_file = out / "results.json"
    if res_file.exists():
        results = json.loads(res_file.read_text())
        diag = float(np.mean([x["target_margin"] for x in results])) if results else None
    gs = run_gate_checks("gate_07_clean_replay_behavior", "Clean text-only replay", [
        ("fresh_process_ran", ok, detail),
        ("target_margin_recorded", diag is not None, f"mean F_target={diag}" if diag is not None else "no results"),
    ], config={"model": ctx.model_id}, write=True, evidence=ctx.evidence)
    return gs


PHASES = [
    ("gate_02_donor_transfer", phase_02_donor),
    ("gate_03a_shared_subspace", phase_03a_extract),
    ("gate_03b_causal_subspace", phase_03b_causal),
    ("gate_04b_candidate_variance", phase_04b_score),
    ("gate_05_selection_reconstruction", phase_05_search),
    ("gate_07_clean_replay_behavior", phase_07_clean_replay),
]


PHASE_BY_ID = {gid: fn for gid, fn in PHASES}


def run_pipeline(config_path: str, smoke: bool = False, model_id: Optional[str] = None,
                 target: str = "eagle", continue_on_fail: bool = False,
                 run_dir: Optional[str] = None, only: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Run the pilot phases in order (or a subset via ``only``).

    ``only`` is a list of gate ids (or short names like '03a','05') to run; prior
    artifacts are restored from ``run_dir`` so phases can run as separate SLURM
    jobs. Without ``only`` the whole pipeline runs in one process.
    """
    cfg = Config.load(config_path)
    fast = smoke or cfg.modes.fast_dev_run
    mid = model_id or (cfg.get("smoke_model_id") if smoke else cfg.get("model.model_id"))
    seed_everything(cfg.get("experiment.seed", 0))
    rd = Path(run_dir or (Path("artifacts/run_manifests") / f"pilot_{'smoke' if smoke else 'full'}_{target}"))
    rd.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(config=cfg, model_id=mid, run_dir=rd, target=target, fast=fast)

    # resolve which phases to run
    phases = PHASES
    if only:
        wanted = set()
        for name in only:
            for gid, _ in PHASES:
                if gid == name or name in gid:
                    wanted.add(gid)
        phases = [(gid, fn) for gid, fn in PHASES if gid in wanted]
        ctx.restore()  # load donors/subspace/context from a prior job

    man = Manifest.create(phase="pilot", model_tag=cfg.get("model.tag", "smoke"), target=target)
    man.model_id = mid
    results: Dict[str, str] = {}
    for gate_id, fn in phases:
        print(f"\n=== {gate_id} ===")
        gs = fn(ctx)
        results[gate_id] = gs.status
        ctx.persist()  # checkpoint artifacts after every phase for resumability
        print(f"--> {gate_id}: {gs.status}")
        if gs.status != PASS and not continue_on_fail:
            print(f"STOP: {gate_id} did not pass ({gs.status}). {gs.failure_summary}")
            break
    man.extra["gate_results"] = results
    man.finalize().save(rd / "pilot_manifest.json")
    return results

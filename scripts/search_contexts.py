#!/usr/bin/env python
"""Selection-only context beam search (§9.4, notebook 08).

Enforces the no-final-eval-import invariant, runs beam search with an
activation-only score, checkpoints every step, and runs matched-null searches.
``--fast`` uses a synthetic additive score so search behavior is exercised
offline (reproducibility, resume, diversity constraints).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import REPO, require_expensive  # noqa: E402
from subliminal_icl.config import Config, RunModes  # noqa: E402
from subliminal_icl import beam_search as bs  # noqa: E402

FORBIDDEN = {"subliminal_icl.evaluation"}


def assert_no_eval_import():
    leaked = FORBIDDEN & set(sys.modules)
    if leaked:
        raise RuntimeError(f"search must not import the evaluator: {sorted(leaked)}")


def fast_demo(cfg: Config):
    assert_no_eval_import()
    rng = np.random.default_rng(0)
    # 30 synthetic candidate rows; "target state" they write ~ hidden weight
    pool = {f"r{i}": float(rng.normal(0, 1)) for i in range(30)}
    source_of = {rid: f"src{int(rid[1:]) % 7}" for rid in pool}  # source prompt ids

    def score_fn(items):
        return float(sum(pool[i] for i in items))  # additive activation-only score

    def propose_fn(beam):
        return list(pool.keys())

    cons = bs.DiversityConstraints(**{
        "forbid_duplicate_ids": True, "forbid_duplicate_source_prompt": True,
        "max_from_one_source": 4, "ngram_block": 0,
    })
    scfg = bs.SearchConfig(beam_width=8, proposal_count=30, max_steps=2, seed=0)
    st = bs.beam_search(score_fn, propose_fn, scfg, constraints=cons, source_of=source_of)
    best = bs.best_beam(st)
    print("selected:", best.items, "score:", round(best.score, 3))

    # reproducibility: same seed -> same beams
    st2 = bs.beam_search(score_fn, propose_fn, scfg, constraints=cons, source_of=source_of)
    assert [b.items for b in st.beams] == [b.items for b in st2.beams], "search not reproducible"

    # resume: save/load state and continue
    path = REPO / "artifacts/candidate_scores/fast_search_state.json"
    st.save(str(path))
    st_loaded = bs.BeamSearchState.load(str(path))
    assert [b.items for b in st_loaded.beams] == [b.items for b in st.beams]

    # matched null: random scores, same budget -> compare best
    def null_score(items):
        return float(sum(rng.normal(0, 1) for _ in items))
    stn = bs.beam_search(null_score, propose_fn, scfg, constraints=cons, source_of=source_of)
    print("null best score:", round(bs.best_beam(stn).score, 3))
    print("reproducible + resumable + null-matched: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/pilot_qwen7b.yaml")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.fast or RunModes.from_env().fast_dev_run:
        fast_demo(cfg)
        return 0
    require_expensive("search_contexts")
    raise SystemExit("Real search scores contexts with the frozen model on GPU; see notebook 08.")


if __name__ == "__main__":
    raise SystemExit(main())

# Parallelism map for the pilot (job 03)

## The honest summary

Within **one trait**, the pilot is mostly a **serial dependency chain** — you
cannot fit the subspace before you have the donors, cannot score candidates
before the subspace, cannot search before scoring, cannot clean-replay before
searching. So the parallelism that actually buys wall-clock time is:

1. **Across traits** — the entire pilot is independent per target trait. This is
   the dominant win. → `03_pilot_array.slurm` (one H200 per trait).
2. **Across search seeds** — the ≥3 required search seeds are independent.
3. Two small **within-trait fan-outs** (see the DAG): the donors and the
   explicit-shift extraction don't depend on each other, and the CoT-carrier
   branch is independent of the number-row search branch.

## Dependency DAG (one trait)

```
teacher-gen(target) ─┐
                     ├─ donor-target ─┐
teacher-gen(neutral)─┤                ├─► d_ft ─┐
                     └─ donor-control ┘          │
                                                 ├─► subspace 3A ─► causal 3B ─┐
base model ───────────► d_sys extraction ────────┘                             │
                                                                               ▼
                       candidate-pool gen ───────────────► candidate-score 4B ─► search 5 ─► clean-replay 6/7 ─► mediation 11 / stats 12
                                                                               │
                                                          CoT carriers 8 ──────┘  (independent branch, joins only at final eval)
```

**Independent nodes that can run at the same time (same trait):**

| group | nodes | why independent |
|-------|-------|-----------------|
| A | `donor-target`  ∥  `donor-control`  ∥  `d_sys extraction` | donors need only their teacher data; d_sys needs only the base model + trait prompts |
| B | `teacher-gen(target)` ∥ `teacher-gen(neutral)` | different system prompts, no shared state |
| C | number-row `search 5` ∥ `CoT carriers 8` | both consume the frozen subspace; neither needs the other |

**Serial critical path (longest chain):**
`teacher-gen(target) → donor-target → d_ft → subspace 3A → causal 3B → score 4B → search 5 → clean-replay 6/7 → mediation 11 → stats 12`

## What the SLURM files give you

### 1. Across-trait parallelism (recommended, works today)
```bash
cd /scratch/pb2276/grad_work/subliminal
sbatch slurm/03_pilot_array.slurm      # --array=0-1; edit TARGETS + range for more traits
```
Each array task is a full, self-contained pilot for one trait on its own H200 —
N traits finish in the wall-clock time of one.

### 2. Resumable per-phase chain (per trait) — `slurm/phases/`
Each phase runs one gate via `run_pilot.py --only <gate> --run-dir <shared>` and
persists/restores artifacts (donors, subspace, candidate rows, context) to the
shared run dir. Chain them with `afterok` so a failed gate stops that trait
(the "stop at each gate" discipline), while other traits keep running.

```bash
cd /scratch/pb2276/grad_work/subliminal
export TARGET=eagle
export RUN_DIR=artifacts/run_manifests/pilot_qwen7b_${TARGET}

a=$(sbatch --parsable --export=ALL,TARGET,RUN_DIR slurm/phases/03a_donor.slurm)
b=$(sbatch --parsable --dependency=afterok:$a --export=ALL,TARGET,RUN_DIR slurm/phases/03b_subspace.slurm)
c=$(sbatch --parsable --dependency=afterok:$b --export=ALL,TARGET,RUN_DIR slurm/phases/03c_causal.slurm)
d=$(sbatch --parsable --dependency=afterok:$c --export=ALL,TARGET,RUN_DIR slurm/phases/03d_score_search.slurm)
sbatch          --dependency=afterok:$d --export=ALL,TARGET,RUN_DIR slurm/phases/03e_clean_replay.slurm
```

Launch one such chain per trait (change `TARGET`/`RUN_DIR`) to run all traits'
chains concurrently. `03c_causal` depends on `03b` but is independent of
`03d_score_search` (both need only the subspace), so you may launch `03c` and
`03d` both with `--dependency=afterok:$b` to run causal validation and the
scoring/search in parallel.

## Finer within-trait fan-out (optional)

Group A (donor-target ∥ donor-control ∥ d_sys) is currently bundled: donors are
trained together in phase `gate_02` and d_sys/d_ft are computed together in
`gate_03a`. To run those three on three GPUs simultaneously you'd split
`gate_02` into two single-donor jobs and pull d_sys out of `gate_03a` into its
own job. The code already persists each artifact independently, so this is a
mechanical split — ask if you want the finer-grained phase files. For most runs
the across-trait array already saturates the cluster, so this extra fan-out
mainly helps when you're validating a single trait as fast as possible.

## Rule of thumb

- Validating the method broadly → **across-trait array** (13 traits → 13 GPUs).
- Iterating on one trait fast → the **per-phase chain**, launching `03c` and
  `03d` in parallel off `03b`, plus the optional Group-A split.
- Statistical power → replicate the array over **≥3 search seeds** (pass a seed
  into the run dir name).

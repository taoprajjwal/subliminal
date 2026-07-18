# SLURM run order — large-scale validation

All jobs use the `gpu2` conda env, the `nvidia` partition, and `-q chi`, matching
the cluster convention. Submit from the repo root (`/scratch/pb2276/grad_work/subliminal`)
so the relative `#SBATCH --output slurm/logs/...` paths resolve.

## Sequence

| # | File | GPU | Time | Gate(s) | What it validates | Status |
|---|------|-----|------|---------|-------------------|--------|
| 0 | `00_env_and_tests.slurm` | 1 (any) | 1h | Gate 0 | import, full pytest (88 tests), all 14 notebooks in FAST_DEV_RUN | **runs today** |
| 1 | `01_data_prep.slurm` | 1 | 4h | Gate 0B | download+normalize eagle dataset, verify arithmetic carriers, build eval splits | **runs today** (paired gen full-path via notebooks) |
| 2 | `02_baseline_qwen7b.slurm` | h200 | 8h | Gate 1 | LIVE evaluator sensitivity/specificity + random-row ICL null on Qwen-7B | **runs today** |
| 3 | `03_pilot_qwen7b.slurm` | h200 | 2d | Gates 2–7 | REAL pipeline: donor → subspace → causal → scoring → search → clean replay | **runs today** (pipeline.py) |
| 3′ | `03_pilot_array.slurm` | h200×N | 2d | Gates 2–7 | same, one array task **per trait** (across-trait parallelism) | **runs today** |
| 3″ | `phases/03a..03e.slurm` | h200 | var | per gate | resumable per-phase chain (see PARALLELISM.md) | **runs today** |
| 4 | `04_replicate_eagle_qwen14b.slurm` | h200 | 4d | replication | mandatory 14B eagle replication | pipeline runs; scale-tune |
| 5 | `05_cross_model_gemma4b.slurm` | h200 | 2d | cross-model | method replication on gemma-3-4b-it (only after gate 3 passes) | pipeline runs; scale-tune |
| 6 | `06_final_report.slurm` | 1 | 30m | report | assemble REPORT.md + standalone clean-replay acceptance (§20) ×2 | **runs today** |

**Parallelism:** the pilot is a serial dependency chain within a trait but
embarrassingly parallel across traits and search seeds. See
[`PARALLELISM.md`](PARALLELISM.md) for the dependency DAG and submission recipes
(`03_pilot_array.slurm` for across-trait; `phases/` for the resumable per-phase
chain, where `03c_causal` and `03d_score_search` can run in parallel off `03b`).

## Submit

```bash
cd /scratch/pb2276/grad_work/subliminal

j0=$(sbatch --parsable slurm/00_env_and_tests.slurm)
j1=$(sbatch --parsable --dependency=afterok:$j0 slurm/01_data_prep.slurm)
j2=$(sbatch --parsable --dependency=afterok:$j1 slurm/02_baseline_qwen7b.slurm)
j3=$(sbatch --parsable --dependency=afterok:$j2 slurm/03_pilot_qwen7b.slurm)
j4=$(sbatch --parsable --dependency=afterok:$j3 slurm/04_replicate_eagle_qwen14b.slurm)
j5=$(sbatch --parsable --dependency=afterok:$j3 slurm/05_cross_model_gemma4b.slurm)
sbatch --dependency=afterok:$j3 slurm/06_final_report.slurm
```

`--dependency=afterok` chains the jobs so a failed gate stops the pipeline
(consistent with the "stop at each gate" discipline). Jobs 4 and 5 both depend on
the 7B pilot (job 3); run them in parallel.

## What "runs today" vs "scaffold" means

- **runs today** — the job's scripts are fully implemented against the live model
  stack and produce real artifacts + gate checkpoints on the GPU node. The pilot
  (`scripts/run_pilot.py` → `src/subliminal_icl/pipeline.py`) was validated
  end-to-end on the 0.5B model: it trains real LoRA donors, extracts d_sys/d_ft,
  fits + causally validates the trait subspace, scores candidate rows, beam-searches
  a selection-only context, and clean-replays it in a fresh import-guarded process.
  Every gate writes PASS/FAIL/INCONCLUSIVE **with model evidence** and stops on FAIL.
- **scale-tune** — the 14B / gemma jobs run the same pipeline; only the config
  sizes (prototype prompt count, candidate pool size, donor epochs, beam width)
  need turning up from the pilot defaults for publication-strength statistics.

Requires `peft` (installed in gpu2). On 0.5B the mechanism gates (subspace 3A,
causal 3B, scoring 4B, search 5) pass; the transfer/behavior gates (2, 7) are
expected to reach significance only at 7B+ with full data volume.

## Extra deps for the pilot job

```bash
conda activate gpu2
pip install peft nbclient nbformat   # peft: donor LoRA; nbclient/nbformat: headless notebook exec
```

## Notes

- CPU-only steps (00, 01, 06) request a generic GPU to satisfy the `nvidia`
  partition; if your cluster exposes a CPU partition, drop `--gres` and switch
  `-p` for those three.
- `HF_HOME` defaults to `/scratch/pb2276/huggingface`; models are cached there.
- Never set `RUN_EXPENSIVE=1` on a login node — only inside these batch jobs.
- Final clean replay runs in a fresh process with an import guard (no hooks /
  adapters); job 6 runs it twice to check reproducibility.

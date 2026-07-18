#!/bin/bash
#SBATCH --job-name=subicl_03_pilot7b
#SBATCH --output=logs/03_pilot7b_%j.out
#SBATCH --error=logs/03_pilot7b_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:h200:1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -q chi

# Single-trait pilot on Qwen2.5-7B, run as ONE process by the real pipeline
# (src/subliminal_icl/pipeline.py): Gate 2 donor -> 3A subspace -> 3B causal ->
# 4B candidate scoring -> 5 selection search -> 6/7 clean replay. Stops at the
# first gate that does not PASS and writes reports/checkpoints/gate_XX.md with
# model evidence. run_pilot.py exits nonzero if any gate fails, so afterok chains
# stop here.
#
# For parallelism (across traits / seeds, and the per-phase resumable chain) see
# slurm/03_pilot_array.slurm, slurm/phases/, and slurm/PARALLELISM.md.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export FAST_DEV_RUN=0
export RUN_EXPENSIVE=1

TARGET="${TARGET:-eagle}"
echo "=== real pilot for target=$TARGET on Qwen2.5-7B ==="
python scripts/run_pilot.py \
  --config configs/experiment/pilot_qwen7b.yaml \
  --target "$TARGET" \
  --run-dir "artifacts/run_manifests/pilot_qwen7b_${TARGET}"

echo "=== Done: 03_pilot_qwen7b ($TARGET) ==="

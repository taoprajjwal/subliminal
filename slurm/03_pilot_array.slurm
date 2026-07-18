#!/bin/bash
#SBATCH --job-name=subicl_03_pilot_arr
#SBATCH --output=logs/03_pilot_arr_%A_%a.out
#SBATCH --error=logs/03_pilot_arr_%A_%a.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:h200:1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -q chi
#SBATCH --array=0-1        # one independent GPU job per target trait (see TARGETS)

# TOP-LEVEL PARALLELISM: the whole pilot is independent per target trait, so we
# run one array task per trait. Each task runs the full serial chain for its
# trait (donor -> subspace -> causal -> score -> search -> clean replay) on its
# own H200. Add traits by extending TARGETS and the --array range.
#
# The per-trait chain is inherently sequential (you cannot score candidates
# before the subspace exists); the parallelism that matters here is ACROSS traits
# and ACROSS seeds. For finer within-trait fan-out see slurm/phases/ and
# slurm/PARALLELISM.md.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export FAST_DEV_RUN=0
export RUN_EXPENSIVE=1

TARGETS=(eagle cat dog wolf owl)      # index by SLURM_ARRAY_TASK_ID
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "=== pilot for target=$TARGET (array task $SLURM_ARRAY_TASK_ID) ==="

python scripts/run_pilot.py \
  --config configs/experiment/pilot_qwen7b.yaml \
  --target "$TARGET" \
  --run-dir "artifacts/run_manifests/pilot_qwen7b_${TARGET}"

echo "=== Done: pilot $TARGET ==="

#!/bin/bash
#SBATCH --job-name=subicl_05_gemma4b
#SBATCH --output=slurm/logs/05_gemma4b_%j.out
#SBATCH --error=slurm/logs/05_gemma4b_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:h200:1
#SBATCH -n 16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH -q chi

# Cross-model replication of the METHOD (not vector transfer) on gemma-3-4b-it
# (§4.3). Run ONLY after the Qwen pilot passes the subspace + clean-replay gates.
# Uses eager attention (gemma attention inspection).
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export FAST_DEV_RUN=0
export RUN_EXPENSIVE=1

echo "=== Gate 1 baseline on gemma-3-4b-it ==="
python scripts/run_baseline_eval.py \
  --model google/gemma-3-4b-it \
  --eval data/splits/animal_preference_eval_dev.jsonl \
  --targets eagle cat \
  --output artifacts/evaluations/baseline_gemma4b

echo "=== cross-model pilot (config resolved + phase plan) ==="
python scripts/run_pilot.py --config configs/experiment/cross_model_gemma4b.yaml || true

echo "=== Done: 05_cross_model_gemma4b ==="

#!/bin/bash
#SBATCH --job-name=subicl_04_rep14b
#SBATCH --output=slurm/logs/04_rep14b_%j.out
#SBATCH --error=slurm/logs/04_rep14b_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:h200:1
#SBATCH -n 16
#SBATCH --mem=160G
#SBATCH --time=4-00:00:00
#SBATCH -q chi

# Mandatory replication on Qwen2.5-14B with the existing eagle-numbers dataset
# (§4.2). No 4-bit quantization for the final activation geometry / clean replay.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export FAST_DEV_RUN=0
export RUN_EXPENSIVE=1

echo "=== Gate 1 baseline on 14B (evaluator + random-row ICL null) ==="
python scripts/run_baseline_eval.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --eval data/splits/animal_preference_eval_dev.jsonl \
  --targets eagle \
  --output artifacts/evaluations/baseline_qwen14b

echo "=== replication pilot (config resolved + phase plan) ==="
python scripts/run_pilot.py --config configs/experiment/replicate_eagle_qwen14b.yaml || true

echo "=== Done: 04_replicate_eagle_qwen14b ==="

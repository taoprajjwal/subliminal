#!/bin/bash
#SBATCH --job-name=subicl_02_baseline
#SBATCH --output=logs/02_baseline_%j.out
#SBATCH --error=logs/02_baseline_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:h200:1
#SBATCH -n 20
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -q chi

# Gate 1 at scale on a LIVE model: evaluator sensitivity/specificity + the
# random-row ICL null. This exercises the whole model stack (loading, chat
# rendering, full-sequence token scoring, confusion matrix, gamma) and writes
# reports/checkpoints/gate_01_evaluator.md. Fully implemented today.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export FAST_DEV_RUN=0
export RUN_EXPENSIVE=1

echo "=== smoke: 0.5B model, 4 prompts (interface check, not science) ==="
python scripts/run_baseline_eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --eval data/splits/animal_preference_eval_dev.jsonl \
  --targets eagle cat --limit 4 \
  --output artifacts/evaluations/baseline_smoke

echo "=== Gate 1: Qwen2.5-7B-Instruct, full eval_dev ==="
python scripts/run_baseline_eval.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --eval data/splits/animal_preference_eval_dev.jsonl \
  --targets eagle cat dog wolf owl \
  --output artifacts/evaluations/baseline_qwen7b

echo "=== Done: 02_baseline_qwen7b ==="

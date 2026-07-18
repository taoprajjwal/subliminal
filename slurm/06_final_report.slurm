#!/bin/bash
#SBATCH --job-name=subicl_06_report
#SBATCH --output=slurm/logs/06_report_%j.out
#SBATCH --error=slurm/logs/06_report_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -q chi

# Assemble reports/final/REPORT.md from status.json + gate checkpoints, and run
# the standalone clean-replay acceptance test (§20) TWICE in fresh processes to
# confirm reproducibility, if a final context has been frozen.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"

echo "=== build final report ==="
python scripts/build_report.py --out reports/final

CTX=artifacts/contexts/final_context.txt
EVAL=data/splits/animal_preference_eval_final.jsonl
if [ -f "$CTX" ]; then
  echo "=== standalone clean replay (§20), run 1 ==="
  python scripts/clean_replay_eval.py --model Qwen/Qwen2.5-7B-Instruct \
    --context "$CTX" --eval "$EVAL" --target eagle \
    --output artifacts/evaluations/final_clean_replay_run1
  echo "=== standalone clean replay (§20), run 2 (fresh process) ==="
  python scripts/clean_replay_eval.py --model Qwen/Qwen2.5-7B-Instruct \
    --context "$CTX" --eval "$EVAL" --target eagle \
    --output artifacts/evaluations/final_clean_replay_run2
else
  echo "no frozen final context yet ($CTX) — skipping clean-replay acceptance test"
  python scripts/clean_replay_eval.py --self-test
fi

echo "=== Done: 06_final_report ==="

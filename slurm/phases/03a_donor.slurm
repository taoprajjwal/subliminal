#!/bin/bash
#SBATCH --job-name=subicl_03a_donor
#SBATCH --output=logs/03a_donor_%j.out
#SBATCH --error=logs/03a_donor_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:h200:1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH -q chi

# Resumable pilot phase. Shares --run-dir with the other phases of the same
# trait; artifacts (donors, subspace, candidate rows, context) are persisted to
# run_dir and restored by --only. TARGET and RUN_DIR come from the environment
# (defaults below) so you can launch one chain per trait.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal
export PYTHONUNBUFFERED=1 PYTHONPATH=src:$PYTHONPATH FAST_DEV_RUN=0 RUN_EXPENSIVE=1
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TARGET="${TARGET:-eagle}"
RUN_DIR="${RUN_DIR:-artifacts/run_manifests/pilot_qwen7b_${TARGET}}"
echo "=== [$TARGET] Gate 2: teacher gen + donor LoRAs (target + control) ==="
python scripts/run_pilot.py --config configs/experiment/pilot_qwen7b.yaml \
  --target "$TARGET" --run-dir "$RUN_DIR" --only gate_02_donor_transfer
echo "=== Done: 03a_donor $TARGET ==="

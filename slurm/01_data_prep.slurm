#!/bin/bash
#SBATCH --job-name=subicl_01_data
#SBATCH --output=logs/01_data_%j.out
#SBATCH --error=logs/01_data_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -q chi

# Gate 0B: download + normalize the existing eagle dataset, generate + verify
# arithmetic carriers, and build the immutable evaluation splits.
# GPU is requested only for the (optional) paired number generation step below.
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
# real download + full-size generation (not FAST_DEV_RUN)
export FAST_DEV_RUN=0

echo "=== download + normalize existing eagle-numbers dataset ==="
python scripts/download_data.py --config configs/data/existing_eagle_numbers.yaml

echo "=== generate + deterministically verify arithmetic carriers ==="
python scripts/generate_arithmetic.py --config configs/data/synthetic_arithmetic.yaml

echo "=== build immutable evaluation splits (eval_dev / eval_final) ==="
python scripts/build_eval_split.py --config configs/data/evaluation.yaml

# Paired number generation under neutral/target/countertrait system prompts uses
# a live model. The generator's full path is driven from notebook 01/03 (which
# expose the exact prompts + top-k logits + divergence annotation). The offline
# fixture below keeps the downstream pipeline runnable without a model:
echo "=== paired number fixture (offline shape) ==="
FAST_DEV_RUN=1 python scripts/generate_paired_numbers.py --fast --target eagle

echo "=== Done: 01_data_prep ==="

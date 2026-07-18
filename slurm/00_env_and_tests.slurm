#!/bin/bash
#SBATCH --job-name=subicl_00_tests
#SBATCH --output=slurm/logs/00_tests_%j.out
#SBATCH --error=slurm/logs/00_tests_%j.err
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -q chi

# Gate 0 at scale: environment, package import, full test suite, notebook smoke.
# Needs a GPU only so the torch-gated hook tests run; drop --gres if your cluster
# has a CPU partition (the numpy tests pass without a GPU).
source ~/.bashrc
conda activate gpu2
cd /scratch/pb2276/grad_work/subliminal

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/pb2276/huggingface}"

echo "=== import check ==="
python -c "import subliminal_icl as s; print('subliminal_icl', s.__version__)"

echo "=== full test suite ==="
python -m pytest -q

echo "=== notebook smoke (FAST_DEV_RUN=1, all 14 notebooks) ==="
FAST_DEV_RUN=1 python scripts/run_notebooks_smoke.py

echo "=== Done: 00_env_and_tests ==="

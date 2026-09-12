#!/bin/bash
#SBATCH -J bench_scaling
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o logs/bench_scaling_%j.out
#SBATCH -e logs/bench_scaling_%j.err

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
python scripts/bench_scaling.py
echo "Done: $(date)"

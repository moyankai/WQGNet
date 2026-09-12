#!/bin/bash
#SBATCH -J bench_eff
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o logs/bench_eff_%j.out
#SBATCH -e logs/bench_eff_%j.err

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

# Wyckoff benchmark
python scripts/bench_efficiency.py \
  --model-type wyckoff \
  --data-dir data/processed/jarvis_lmdb_bandgap \
  --checkpoint results/train-jarvis_leaderboard-128x0e_4x1o-L5-optb88vdw_bandgap-s2-20260807-1650/best.pt \
  --out results/bench_efficiency/wyckoff.json

# P1 matched benchmark (random init, no checkpoint)
python scripts/bench_efficiency.py \
  --model-type p1 \
  --data-dir data/processed/jarvis_lmdb_bandgap_p1_std \
  --out results/bench_efficiency/p1.json

echo "Done: $(date)"

#!/bin/bash
#SBATCH -J correctness_suite
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -o logs/correctness_suite_%j.out
#SBATCH -e logs/correctness_suite_%j.err

# Run the Section 2.3 correctness suite on the full JARVIS dataset.
#
# Usage:
#   sbatch jobs/correctness_suite.slurm
#   sbatch jobs/correctness_suite.slurm -- --benchmark-limit 100   # quick test
#
# Results are written to results/correctness_suite/<timestamp>/
# and the LaTeX macro file is updated automatically.

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

python scripts/run_correctness_suite.py \
    --parts 1,2,3,4 \
    --benchmark-limit -1 \
    --precision float32 \
    --cutoff 5.0 \
    "$@"

echo "Done: $(date)"

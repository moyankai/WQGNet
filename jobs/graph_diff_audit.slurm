#!/bin/bash
#SBATCH -J graph_diff_audit
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 5-00:00:00
#SBATCH -o logs/graph_diff_audit_%j.out
#SBATCH -e logs/graph_diff_audit_%j.err

# Full-data graph-diff audit: OLD (stored LMDB) vs FIXED (fresh rebuild).
# Parallelized with multiprocessing over 28 CPU cores.
#
# Usage:
#   sbatch jobs/graph_diff_audit.slurm

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs results/graph_diff_audit

echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Start: $(date)"

python scripts/run_graph_diff_audit.py \
    --num-workers 28 \
    --output-dir results/graph_diff_audit

echo "Done: $(date)"

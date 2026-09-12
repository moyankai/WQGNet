#!/bin/bash
#SBATCH -J correctness_part2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -o logs/correctness_part2_%j.out
#SBATCH -e logs/correctness_part2_%j.err

# Run Part 2 (quotient-vs-P1 equivalence) on the full JARVIS dataset.
# This is a follow-up to the main correctness_suite job.

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

python scripts/run_correctness_suite.py \
    --parts 2 \
    --benchmark-limit -1 \
    --precision float32 \
    --cutoff 5.0 \
    --existing-timestamp 20260718_194550 \
    --no-assemble \
    "$@"

echo "Part 2 done: $(date)"
echo "Now assemble with:"
echo "  python -c \"from tests.correctness_suite.assemble_table1 import assemble; assemble('results/correctness_suite/20260718_194550', 'docs/WyckoffGNN_v6_2_Overleaf/WyckoffGNN_v6_2_Overleaf/tab_consistency_values.tex')\""

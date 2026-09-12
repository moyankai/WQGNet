#!/bin/bash
#SBATCH -J benchmark
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o logs/staging_benchmark_%j.out
#SBATCH -e logs/staging_benchmark_%j.err

# Run multi-model multi-seed benchmark.
#
# Usage:
#   sbatch jobs/benchmark.slurm                         # default config
#   sbatch jobs/benchmark.slurm configs/benchmark.yaml  # explicit config

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

CONFIG=${1:-configs/benchmark.yaml}
shift 2>/dev/null || true

OUTPUT_ROOT=$(wyckoffgnn resolve-output-dir --config "$CONFIG" --command benchmark "$@")
mkdir -p "$OUTPUT_ROOT"

exec > "$OUTPUT_ROOT/slurm_job.out" 2> "$OUTPUT_ROOT/slurm_job.err"

echo "Config:      $CONFIG"
echo "Output root: $OUTPUT_ROOT"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:       $(date)"

wyckoffgnn benchmark --config "$CONFIG" --output-root "$OUTPUT_ROOT" "$@"

echo "Done: $(date)"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    for suffix in out err; do
        stage="logs/staging_benchmark_${SLURM_JOB_ID}.${suffix}"
        [ -f "$stage" ] && mv "$stage" "$OUTPUT_ROOT/slurm_startup.${suffix}" || true
    done
fi

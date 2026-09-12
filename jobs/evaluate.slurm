#!/bin/bash
#SBATCH -J evaluate
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o logs/staging_evaluate_%j.out
#SBATCH -e logs/staging_evaluate_%j.err

# Evaluate a trained checkpoint on test set.
#
# Usage:
#   sbatch jobs/evaluate.slurm CONFIG CHECKPOINT [extra --override ...]
#
# Example:
#   sbatch jobs/evaluate.slurm configs/evaluate.yaml results/.../best.pt

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

CONFIG=${1:?Usage: sbatch jobs/evaluate.slurm CONFIG CHECKPOINT}
CHECKPOINT=${2:?Usage: sbatch jobs/evaluate.slurm CONFIG CHECKPOINT}
shift 2

OUTPUT_DIR=$(wyckoffgnn resolve-output-dir --config "$CONFIG" --command eval "$@")
mkdir -p "$OUTPUT_DIR"

exec > "$OUTPUT_DIR/slurm_job.out" 2> "$OUTPUT_DIR/slurm_job.err"

echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Start:      $(date)"

wyckoffgnn evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "Done: $(date)"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    for suffix in out err; do
        stage="logs/staging_evaluate_${SLURM_JOB_ID}.${suffix}"
        [ -f "$stage" ] && mv "$stage" "$OUTPUT_DIR/slurm_startup.${suffix}" || true
    done
fi

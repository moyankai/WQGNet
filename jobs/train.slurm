#!/bin/bash
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
# Slurm can't reference variables in -o/-e paths, so we start with a temp
# staging log and move it into the result folder once the path is known.
#SBATCH -o logs/staging_train_%j.out
#SBATCH -e logs/staging_train_%j.err

# Train a WyckoffGNN model on preprocessed shard cache.
#
# All outputs — including this job's stdout/stderr — are routed into a
# per-run result folder under results/. The staging log configured above
# is used only briefly at startup and gets absorbed into the run folder.
#
# Usage:
#   sbatch jobs/train.slurm                         # default config
#   sbatch jobs/train.slurm configs/train.yaml      # explicit config
#
# Example with overrides (bandgap):
#   sbatch jobs/train.slurm configs/train.yaml \
#     --override "property_name=optb88vdw_bandgap" \
#                "manifest_path=data/processed/jarvis_wyckoff_lmdb/manifest.jsonl" \
#                "shard_dir=data/processed/jarvis_wyckoff_lmdb" \
#                "use_edge_state=true" "edge_readout=true" "batch_size=128"

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

CONFIG=${1:-configs/train.yaml}
shift 2>/dev/null || true

# Resolve the target result folder from the config + overrides.
OUTPUT_DIR=$(wyckoffgnn resolve-output-dir --config "$CONFIG" --command train "$@")
mkdir -p "$OUTPUT_DIR"

# Redirect the rest of this job (and its child processes) into the run folder.
exec > "$OUTPUT_DIR/slurm_job.out" 2> "$OUTPUT_DIR/slurm_job.err"

echo "Config:     $CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: $@"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:      $(date)"

wyckoffgnn train --config "$CONFIG" --output-dir "$OUTPUT_DIR" "$@"

echo "Done: $(date)"

# Absorb the pre-redirect staging log (contains conda activation etc.).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    for suffix in out err; do
        stage="logs/staging_train_${SLURM_JOB_ID}.${suffix}"
        [ -f "$stage" ] && mv "$stage" "$OUTPUT_DIR/slurm_startup.${suffix}" || true
    done
fi

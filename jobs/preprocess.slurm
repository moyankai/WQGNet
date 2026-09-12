#!/bin/bash
#SBATCH -J preprocess
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -o logs/staging_preprocess_%j.out
#SBATCH -e logs/staging_preprocess_%j.err

# Preprocess raw dataset into LMDB shard cache.
#
# Slurm stdout/stderr are redirected into the LMDB cache directory so all
# preprocess artefacts stay next to each other.
#
# Usage:
#   sbatch jobs/preprocess.slurm                         # default config
#   sbatch jobs/preprocess.slurm configs/preprocess.yaml # explicit config

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

CONFIG=${1:-configs/preprocess.yaml}
shift 2>/dev/null || true

# Resolve cache_dir accounting for CLI overrides.
CACHE_DIR=$(python -c "
import sys
from wyckoff_gnn.utils.config import load_config
overrides = sys.argv[1:]
# Filter out --override flag and grab only key=value pairs after it.
ov = []
i = 0
while i < len(overrides):
    if overrides[i] == '--override':
        i += 1
        while i < len(overrides) and not overrides[i].startswith('--'):
            ov.append(overrides[i]); i += 1
    else:
        i += 1
cfg = load_config('$CONFIG', overrides=ov)
print((cfg.get('preprocess') or {}).get('cache_dir', 'data/processed/default'))
" "$@")
mkdir -p "$CACHE_DIR"

exec > "$CACHE_DIR/slurm_job.out" 2> "$CACHE_DIR/slurm_job.err"

echo "Config:    $CONFIG"
echo "Cache dir: $CACHE_DIR"
echo "Extra:     $@"
echo "Start:     $(date)"

wyckoffgnn preprocess --config "$CONFIG" "$@"

echo "Done: $(date)"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    for suffix in out err; do
        stage="logs/staging_preprocess_${SLURM_JOB_ID}.${suffix}"
        [ -f "$stage" ] && mv "$stage" "$CACHE_DIR/slurm_startup.${suffix}" || true
    done
fi

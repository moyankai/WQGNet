#!/bin/bash
#SBATCH -J nequip_bg
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o logs/staging_nequip_bg_%j.out
#SBATCH -e logs/staging_nequip_bg_%j.err

# Run NequIP baseline on JARVIS bandgap.
#
# Usage:
#   sbatch jobs/nequip_train.slurm                                       # default
#   sbatch jobs/nequip_train.slurm configs/nequip_jarvis_bandgap.yaml    # explicit

set -eo pipefail
cd /public/home/moyk/wyckoff_gnn
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate nequip_bench
export PYTHONUNBUFFERED=1
mkdir -p logs

CONFIG=${1:-configs/nequip_jarvis_bandgap.yaml}
shift 2>/dev/null || true

CONFIG_DIR=$(dirname "$CONFIG")
CONFIG_NAME=$(basename "$CONFIG")
CONFIG_NAME=${CONFIG_NAME%.yaml}

echo "Config: $CONFIG"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start:  $(date)"

nequip-train --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME" "$@"

echo "Done:   $(date)"

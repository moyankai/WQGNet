#!/bin/bash
#SBATCH -J screen_redundancy
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 8:00:00
#SBATCH -o logs/screen_redundancy_%j.out
#SBATCH -e logs/screen_redundancy_%j.err

set -eo pipefail
cd /data-storage/home/huwei/moyk/wyckoff_gnn
source /data-storage/software/Anaconda/anaconda3/etc/profile.d/conda.sh
conda activate wyckoff_gnn
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Start: $(date)"

SOURCE=${1:-both}
shift 2>/dev/null || true

python -u scripts/screen_crystallographic_redundancy.py \
    --source "$SOURCE" --workers "${SLURM_NTASKS:-16}" "$@"

echo "Done: $(date)"

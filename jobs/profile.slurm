#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/moyk/wyckoff_gnn/results/profile_out.txt
#SBATCH --error=/public/home/moyk/wyckoff_gnn/results/profile_err.txt

cd /public/home/moyk/wyckoff_gnn
source ~/.bashrc
conda activate wyckoff_gnn
python profile_forward.py

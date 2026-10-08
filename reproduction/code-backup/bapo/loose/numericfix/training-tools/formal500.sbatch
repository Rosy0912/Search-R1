#!/bin/bash
#SBATCH --job-name=bapo-rfix-formal500-numeric
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --export=NONE
#SBATCH --chdir=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric/logs/slurm-formal500-%j.out
#SBATCH --error=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric/logs/slurm-formal500-%j.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --kill-on-bad-exit=1 bash /dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric/tools/stage.sh formal500

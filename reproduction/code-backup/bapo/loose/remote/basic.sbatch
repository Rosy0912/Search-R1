#!/bin/bash
#SBATCH --job-name=bapo-hw-basic
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --export=NONE
#SBATCH --chdir=/dataset/shaoshitong/search-r1-rl/bapo
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/bapo/logs/slurm-basic-%j.out
#SBATCH --error=/dataset/shaoshitong/search-r1-rl/bapo/logs/slurm-basic-%j.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --kill-on-bad-exit=1 bash /dataset/shaoshitong/search-r1-rl/bapo/tools/stage.sh hardware-basic

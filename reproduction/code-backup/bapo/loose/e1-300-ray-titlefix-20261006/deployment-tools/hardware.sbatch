#!/bin/bash
#SBATCH --job-name=bapo-instruct300-hardware
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --export=NONE
#SBATCH --chdir=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006/logs/slurm-hardware-%j.out
#SBATCH --error=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006/logs/slurm-hardware-%j.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --kill-on-bad-exit=1 bash /dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006/tools/stage.sh hardware

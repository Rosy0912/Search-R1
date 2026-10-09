#!/bin/bash
#SBATCH --job-name=bapo-aligned-em200-node11
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=512G
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009
#SBATCH --output=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009/logs/slurm-formal-%j.out
#SBATCH --error=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009/logs/slurm-formal-%j.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --kill-on-bad-exit=1 bash /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009/tools/deployment/cohost-node11-v2/formal_worker.sh "${1:?token}"

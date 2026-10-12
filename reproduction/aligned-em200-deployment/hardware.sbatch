#!/bin/bash
#SBATCH --job-name=bapo-aligned-em200-hardware
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009
#SBATCH --output=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/logs/slurm-hardware-%j.out
#SBATCH --error=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/logs/slurm-hardware-%j.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --kill-on-bad-exit=1 bash /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/tools/deployment/hardware_worker.sh "${1:?token required}"

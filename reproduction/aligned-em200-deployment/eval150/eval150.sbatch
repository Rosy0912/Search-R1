#!/bin/bash
#SBATCH --job-name=bapo-eval150-final51713
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/logs/slurm-eval150-%j.out
#SBATCH --error=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/logs/slurm-eval150-%j.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --kill-on-bad-exit=1 bash /dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/tools/eval150/worker.sh "${1:?token}"

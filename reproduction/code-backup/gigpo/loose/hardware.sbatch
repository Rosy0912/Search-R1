#!/bin/bash
#SBATCH --job-name=gigpo-hardware
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo/logs/hardware-%j.log
set -euo pipefail
srun --ntasks=1 bash /mnt/data/shaoshitong/search-r1-rl/gigpo/tools/hardware_body.sh

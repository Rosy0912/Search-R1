#!/bin/bash
#SBATCH --job-name=gigpo-native-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo/logs/native-cpu-%j.log
set -euo pipefail
srun --ntasks=1 bash /mnt/data/shaoshitong/search-r1-rl/gigpo/tools/native_cpu_body.sh

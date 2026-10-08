#!/bin/bash
#SBATCH --job-name=gigpo-e1-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/logs/native-%j.log
set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ROOT/env/bin:$PATH" PYTHONPATH="$ROOT/repo:$ROOT/tools:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/test-lib"
cd "$ROOT/repo"
srun --ntasks=1 "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/native_run.py" "$1"

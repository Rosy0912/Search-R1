#!/bin/bash
#SBATCH --job-name=gigpo-e1-fresh75-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/logs/fresh75-native-%j.log
set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004
export GIGPO_RESUME_REPO="$2"
HELPERS="$3"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ROOT/env/bin:$PATH" PYTHONPATH="$GIGPO_RESUME_REPO:$HELPERS:$ROOT/tools:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/test-lib"
cd "$GIGPO_RESUME_REPO"
srun --ntasks=1 "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$HELPERS/native_run_fresh_restore_v2.py" "$1"

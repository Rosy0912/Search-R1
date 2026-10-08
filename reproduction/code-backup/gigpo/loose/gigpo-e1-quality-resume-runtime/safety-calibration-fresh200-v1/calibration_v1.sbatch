#!/bin/bash
#SBATCH --job-name=gigpo-e1-fresh200-diagnostic2
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/logs/fresh200-diagnostic2-%j.log
set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006
export GIGPO_RESUME_REPO="$3"
HELPERS="$4"
ENVIRONMENT_ROOT="$5"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ENVIRONMENT_ROOT/bin:$PATH" PYTHONPATH="$GIGPO_RESUME_REPO:$HELPERS:$ROOT/tools"
cd "$GIGPO_RESUME_REPO"
srun --ntasks=1 "$ENVIRONMENT_ROOT/bin/python" -m search_r1.baselines.preflight -- "$ENVIRONMENT_ROOT/bin/python" "$HELPERS/calibration_run_v1.py" "$1" "$2"

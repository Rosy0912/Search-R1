#!/bin/bash
#SBATCH --job-name=gigpo-quality512-select
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
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/logs/quality512-select-%j.log
set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004
export GIGPO_RESUME_REPO="$3"
HELPERS="$4"
export GIGPO_EVAL_CANDIDATES_FILE="$HELPERS/candidates.json"
export GIGPO_EVAL_SUMMARY="$ROOT/logs/quality512-selection-job${SLURM_JOB_ID}.json"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ROOT/env/bin:$PATH" PYTHONPATH="$GIGPO_RESUME_REPO:$HELPERS:$ROOT/tools"
cd "$GIGPO_RESUME_REPO"
srun --ntasks=1 "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$HELPERS/eval_candidates_v1.py" --stage quality512 --binding "$1" --binding-sha "$2"

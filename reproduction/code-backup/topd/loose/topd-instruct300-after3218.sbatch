#!/bin/bash
#SBATCH --job-name=topd-instruct300
#SBATCH --nodelist=h100-11
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-instruct300-%j.log
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=7-00:00:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/start_topd_after_teacher3218.py"

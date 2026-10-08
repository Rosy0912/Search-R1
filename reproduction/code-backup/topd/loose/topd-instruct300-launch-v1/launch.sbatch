#!/bin/bash
#SBATCH --job-name=topd-instruct300
#SBATCH --nodelist=h100-11
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-instruct300-%j.log
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=7-00:00:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export AGENT_ROOT="$TASK_ROOT" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_ROOT/repo-terminal-fixed413:$TASK_ROOT/tools/topd-instruct300-launch-v1/runtime"
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
: "${TOPD_CONFIG:?Actual teacher100 bound student configuration required}"
exec "$TASK_ROOT/env/bin/python" "$TASK_ROOT/repo-terminal-fixed413/search_r1/baselines/preflight.py" -- "$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/topd-instruct300-launch-v1/runtime/supervise.py" --config "$TOPD_CONFIG"

#!/bin/bash
#SBATCH --job-name=topd-shared32b-owned
#SBATCH --nodelist=h100-11
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-shared32b-launch-%j.log
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=3-00:00:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export AGENT_ROOT="$TASK_ROOT" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TASK_ROOT/repo-shared32b:$TASK_ROOT/tools/topd-shared32b-supervisor-r1-v2"
: "${TOPD32B_REVIEWED_CONFIG:?Exact independent reviewed config required}"
: "${TOPD32B_EXECUTION_MODE:?profile or formal500 required}"
exec "$TASK_ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/topd-shared32b-supervisor-r1-v2/supervise.py" --config "$TOPD32B_REVIEWED_CONFIG" --mode "$TOPD32B_EXECUTION_MODE"

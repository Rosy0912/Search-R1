#!/bin/bash
#SBATCH --job-name=r1-32b-grpo
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/r1-32b-grpo-%j.log
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=7-00:00:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${R1_CONFIG:?bound reviewed static config}"
: "${R1_TOOLS:?frozen tool directory}"
: "${R1_PREFLIGHT:?fixed lease preflight}"
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$TASK_ROOT/env/bin/python" "$R1_PREFLIGHT" -- "$TASK_ROOT/env/bin/python" "$R1_TOOLS/supervise.py" --config "$R1_CONFIG"

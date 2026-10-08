#!/bin/bash
#SBATCH --job-name=r1-stream-adamw-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/r1-stream-adamw-cpu-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${R1_STREAM_TESTS:?frozen contract directory}"
: "${R1_STREAM_SOURCE:?new frozen source directory}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_ROOT/test-tools"
exec "$TASK_ROOT/env/bin/python" "$R1_STREAM_TESTS/native_entry.py"

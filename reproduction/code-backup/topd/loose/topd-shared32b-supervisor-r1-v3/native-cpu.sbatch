#!/bin/bash
#SBATCH --job-name=topd32b-supervisor-cpu
#SBATCH --nodelist=h100-11
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-shared32b-source-cpu-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export AGENT_ROOT="$TASK_ROOT" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_ROOT/test-tools:$TASK_ROOT/tools/topd-shared32b-supervisor-r1-v3"
OUTPUT="$TASK_ROOT/logs/topd-shared-supervisor-cpu-$SLURM_JOB_ID"
mkdir -p "$OUTPUT"
test "$(git -C "$TASK_ROOT/repo-shared32b" rev-parse HEAD)" = d23c56c78fdb09f40b7859e02b14eeddbe83a863
"$TASK_ROOT/env/bin/python" -c 'import torch; assert not torch.cuda.is_initialized(); print("CPU_ONLY_NO_CUDA_CONTEXT", torch.__version__)'
"$TASK_ROOT/env/bin/python" -m pytest -q "$TASK_ROOT/tools/topd-shared32b-supervisor-r1-v3/test_contracts.py" --basetemp="$OUTPUT/pytest-temp" -o cache_dir="$OUTPUT/pytest-cache"

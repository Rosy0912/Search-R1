#!/bin/bash
#SBATCH --job-name=r1-32b-source-cpu
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/r1-32b-source-cpu-%j.log
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${R1_TOOLS:?frozen tool directory}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$R1_TOOLS:$TASK_ROOT/test-tools"
export R1_SOURCE="$TASK_ROOT/tools/search-r1-32b-grpo100-source-v2"
export R1_CPU_RESULT="$TASK_ROOT/logs/r1-32b-source-cpu-$SLURM_JOB_ID"
exec "$TASK_ROOT/env/bin/python" "$R1_TOOLS/native_entry.py"

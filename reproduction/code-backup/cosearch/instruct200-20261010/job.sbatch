#!/usr/bin/env bash
#SBATCH --job-name=cosearch-instruct200
#SBATCH --partition=acd_ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=7-00:00:00
set -euo pipefail
COSEARCH_TASK_BASE=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
export COSEARCH_RUN_ROOT="$COSEARCH_TASK_BASE/runtime/cosearch-instruct200-common-em-20261011"
export PYTHONPATH="$COSEARCH_RUN_ROOT/tools"
export PYTHONUNBUFFERED=1
cd "$COSEARCH_RUN_ROOT"
exec "$COSEARCH_TASK_BASE/source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11" "$COSEARCH_RUN_ROOT/tools/run_allocation.py"

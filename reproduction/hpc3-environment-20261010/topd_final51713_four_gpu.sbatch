#!/bin/bash
#SBATCH --job-name=topd-final51713
#SBATCH --account=mou176
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/runtime/logs/topd-final51713-%j.log
set -euo pipefail
source /data/user/jxu957/experiments/topd-20261008/environment/activate-topd.sh
TASK_ROOT=/data/user/jxu957/experiments/topd-20261008/runtime
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TASK_ROOT/tools/topd-instruct300-source-v1"
export HF_DATASETS_CACHE="$TASK_ROOT/logs/topd-final51713-recovery-707690/cache/datasets"
export TMPDIR="$TASK_ROOT/logs/topd-final51713-recovery-707690/cache/tmp"
mkdir -p "$TMPDIR"
exec "$TASK_ROOT/env/bin/python" -B "$TASK_ROOT/tools/topd-final51713-v2-20261012/topd_final51713_four_gpu.py"

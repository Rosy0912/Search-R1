#!/bin/bash
#SBATCH --job-name=topd-stage512-four-gpu
#SBATCH --account=mou176
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/runtime/logs/topd-stage512-%j.log
set -euo pipefail
source /data/user/jxu957/experiments/topd-20261008/environment/activate-topd.sh
TASK_ROOT=/data/user/jxu957/experiments/topd-20261008/runtime
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TASK_ROOT/tools/topd-instruct300-source-v1"
export HF_DATASETS_CACHE="$TASK_ROOT/logs/topd-stage512-four-gpu-$SLURM_JOB_ID/cache/datasets"
export TMPDIR="$TASK_ROOT/logs/topd-stage512-four-gpu-$SLURM_JOB_ID/cache/tmp"
mkdir -p "$TMPDIR"
exec "$TASK_ROOT/env/bin/python" -B "$TASK_ROOT/tools/topd-stage512-four-gpu-v2-20261012/topd_stage512_four_gpu_v2.py"

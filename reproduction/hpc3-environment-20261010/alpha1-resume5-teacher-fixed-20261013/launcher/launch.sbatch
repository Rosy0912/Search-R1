#!/bin/bash
#SBATCH --job-name=topd-alpha1-200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=mou176
#SBATCH --partition=acd_u
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/runtime/logs/topd-instruct300-%j.log
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=7-00:00:00
set -euo pipefail
source /data/user/jxu957/experiments/topd-20261008/environment/activate-topd.sh
TASK_ROOT=/data/user/jxu957/experiments/topd-20261008/runtime
export AGENT_ROOT="$TASK_ROOT" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_ROOT/repo-terminal-fixed413:$TASK_ROOT/tools/topd-alpha1-launch-resume5-20261013/runtime"
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
: "${TOPD_CONFIG:?Prepared user-approved teacher85 HPC3 configuration required}"
RUN_CONFIG="$TASK_ROOT/logs/topd-hpc3-${SLURM_JOB_ID}/config.json"
"$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/topd-alpha1-launch-resume5-20261013/runtime/bind_hpc3_run.py" --template "$TOPD_CONFIG" --output "$RUN_CONFIG"
TOPD_CONFIG="$RUN_CONFIG"
exec "$TASK_ROOT/env/bin/python" "$TASK_ROOT/repo-terminal-fixed413/search_r1/baselines/preflight.py" -- "$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/topd-alpha1-launch-resume5-20261013/runtime/supervise.py" --config "$TOPD_CONFIG"

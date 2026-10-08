#!/bin/bash
#SBATCH --job-name=topd32b-score-caller-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$TASK_ROOT/test-tools:$TASK_ROOT/repo-terminal-fixed413"
cd "$TASK_ROOT/tools/topd-32b-native-score-gate-v4"
sha256sum -c files.sha256
TASK_RUN="$TASK_ROOT/logs/topd32b-score-caller-cpu-$SLURM_JOB_ID"
mkdir "$TASK_RUN"
"$TASK_ROOT/env/bin/python" -m pytest -q --basetemp "$TASK_RUN/tmp" -o "cache_dir=$TASK_RUN/cache" test_score_gate.py test_caller_contracts.py
printf 'NATIVE_SCORE_CALLER_POLICY41_PASS_NO_MODEL_NO_GPU_NO_QUALITY\n'

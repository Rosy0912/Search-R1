#!/bin/bash
#SBATCH --job-name=standard-r1-teacher-identity-v4-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/standard-r1-teacher-identity-v4-cpu-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
PACKAGE="$TASK_ROOT/tools/standard-search-r1-teacher-identity-v4"
RUN="$TASK_ROOT/logs/standard-r1-teacher-identity-v4-cpu-$SLURM_JOB_ID"
mkdir "$RUN"
export AGENT_ROOT="$TASK_ROOT" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TASK_ROOT/test-tools:$TASK_ROOT/repo-shared32b"
cd "$PACKAGE"
"$TASK_ROOT/env/bin/python" "$PACKAGE/native_cpu_runner.py" "$PACKAGE" "$RUN"

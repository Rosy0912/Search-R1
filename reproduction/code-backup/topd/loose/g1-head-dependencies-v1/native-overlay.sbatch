#!/bin/bash
#SBATCH --job-name=g1-headdeps-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
TASK_TOOLS="$TASK_ROOT/tools/g1-head-dependencies-v1"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$TASK_ROOT/test-tools"
TASK_RUN="$TASK_ROOT/logs/g1-head-deps-job-$SLURM_JOB_ID"
mkdir "$TASK_RUN"
"$TASK_ROOT/env/bin/python" -m pytest -q --basetemp "$TASK_RUN/pytest-tmp" -o "cache_dir=$TASK_RUN/pytest-cache" "$TASK_TOOLS/test_install.py" > "$TASK_RUN/pytest.log"
exec "$TASK_ROOT/env/bin/python" "$TASK_TOOLS/install.py" --files-sha256 "$1"

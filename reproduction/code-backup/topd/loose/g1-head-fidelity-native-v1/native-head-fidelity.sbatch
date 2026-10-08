#!/bin/bash
#SBATCH --job-name=g1-headfit-fidelity-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$TASK_ROOT/test-tools"
exec "$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/g1-head-fidelity-native-v1/native_run.py" --files-sha256 "$1" --install-receipt-sha256 "$2"

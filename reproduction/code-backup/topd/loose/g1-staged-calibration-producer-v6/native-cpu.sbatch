#!/bin/bash
#SBATCH --job-name=g1-staged-calibration-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
TASK_TOOLS="$TASK_ROOT/tools/g1-staged-calibration-producer-v6"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_ROOT/test-tools${PYTHONPATH:+:$PYTHONPATH}"
exec "$TASK_ROOT/env/bin/python" "$TASK_TOOLS/native_cpu.py" --manifest-sha256 "$1"

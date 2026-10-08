#!/bin/bash
#SBATCH --job-name=r1-hf-semantic-v5-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/r1-hf-semantic-v5-cpu-%j.log
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$ROOT/env/bin/python" -B "$ROOT/tools/r1-hf-semantic-v5-native-operation/entry.py" --request "$R1_HF_REQUEST" --output "$R1_HF_OUTPUT"

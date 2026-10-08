#!/bin/bash
#SBATCH --job-name=topd-quality64-source-recompute
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/quality64-source-recompute-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TASK_ROOT/repo-terminal-fixed413" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ "${SLURM_GPUS_ON_NODE:-0}" != 0 ]; then exit 78; fi
test "$#" -eq 2
"$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/qwen25-32b-quality64-review/review_quality.py" --evaluation "$1" --binding "$TASK_ROOT/logs/qwen25-32b-terminal-fixed413-binding.json" --repo "$TASK_ROOT/repo-terminal-fixed413" --output "$2"

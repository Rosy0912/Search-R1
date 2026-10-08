#!/bin/bash
#SBATCH --job-name=g1-initial-source-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${PIPELINE_ROOT:?frozen pipeline directory}"
: "${CANDIDATE_ROOT:?frozen v3 source directory}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TASK_ROOT/test-tools:$PIPELINE_ROOT:$CANDIDATE_ROOT"
OUT="$TASK_ROOT/logs/g1-pipeline-native-cpu-$SLURM_JOB_ID"
mkdir -p "$OUT"
"$TASK_ROOT/env/bin/python" -m pytest -q "$PIPELINE_ROOT/test_pipeline.py" "$PIPELINE_ROOT/test_guardian.py" "$PIPELINE_ROOT/test_retrieval_gate.py" "$CANDIDATE_ROOT/reproduction/conditional-teacher100/test_conditional_candidate.py" --basetemp "$OUT/pytest-temp" -o "cache_dir=$OUT/pytest-cache"

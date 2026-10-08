#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo-retrieval-h100
test "${SLURM_JOB_NODELIST:?}" = h100-11
test -n "${CUDA_VISIBLE_DEVICES:?}"
export PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="$ROOT/cache/datasets" XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$ROOT/cache" "$ROOT/runs"
cd "$ROOT/repo"
git diff --quiet f4f457df55fbfe99691d466d5c6f1d3fd7175c65 -- search_r1 verl
exec "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/h100_gpu_retrieval_service.py" --run "h100-retrieval-$SLURM_JOB_ID" --baseline "$ROOT/inputs/retrieval-cpu-2854.json" --max-seconds 14400

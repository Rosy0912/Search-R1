#!/bin/bash
set -euo pipefail
BASE=/data/search-r1
exec 9>"$BASE/gpu-job.lock"
flock -n 9
unset PYTHONPATH PYTHONHOME RAY_ADDRESS
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=8
cd "$BASE/repo"
python reproduction/a800/gpu_idle.py 5 6
timeout 600 env CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH="$BASE/retriever-gpu-lib:$BASE/repo" \
    python reproduction/a800/gpu_retrieval_preflight_a800.py
python reproduction/a800/gpu_idle.py 0
timeout 300 env CUDA_VISIBLE_DEVICES=0 python reproduction/h20/model_smoke.py
echo A800_CURRENT_PREFLIGHT_COMPLETE

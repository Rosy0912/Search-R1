#!/usr/bin/env bash
set -euo pipefail
export RECURSIVE_BRAN_BASE=/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011
cd "${RECURSIVE_BRAN_SOURCE:?explicit immutable source required}"
python=/home/shaoshitong/search-e1-config-study-20261004/env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3 RECURSIVE_BRAN_HARDWARE=hkust5 RECURSIVE_BRAN_STRICT_GPU_GATE=1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export RAY_ADDRESS=local
export RECURSIVE_BRAN_EXPERIMENT_LOCK=/home/shaoshitong/search-e1-config-study-20261004/experiment.lock
export TMPDIR="$RECURSIVE_BRAN_BASE/tmp" RAY_TMPDIR=/home/shaoshitong/rbb5-ray XDG_CACHE_HOME="$RECURSIVE_BRAN_BASE/cache"
export HF_HOME="$RECURSIVE_BRAN_BASE/cache/hf"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"
exec "$python" -m search_r1.baselines.preflight -- "$python" release/recursive_bran_b/train_hkust5.py

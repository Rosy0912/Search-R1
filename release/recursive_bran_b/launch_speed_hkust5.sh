#!/usr/bin/env bash
set -euo pipefail
export RECURSIVE_BRAN_BASE=/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011
cd "${RECURSIVE_BRAN_SOURCE:?explicit immutable source}"
python=/home/shaoshitong/search-e1-config-study-20261004/env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3 RECURSIVE_BRAN_HARDWARE=hkust5 RECURSIVE_BRAN_STRICT_GPU_GATE=1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline RAY_ADDRESS=local
export RECURSIVE_B_SKIP_ONLY_REFUSED_OPPORTUNITY=1
export RECURSIVE_BRAN_EXPERIMENT_LOCK=/home/shaoshitong/search-e1-config-study-20261004/experiment.lock
export TMPDIR="$RECURSIVE_BRAN_BASE/tmp" RAY_TMPDIR=/home/shaoshitong/rbb5-speed-ray XDG_CACHE_HOME="$RECURSIVE_BRAN_BASE/cache" HF_HOME="$RECURSIVE_BRAN_BASE/cache/hf"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"
exec "$python" -B -m search_r1.baselines.preflight -- "$python" -B release/recursive_bran_b/resume_hkust5.py --record "${RECURSIVE_BRAN_RESUME_RECORD:?}" --record-sha256 "${RECURSIVE_BRAN_RESUME_RECORD_SHA256:?}"

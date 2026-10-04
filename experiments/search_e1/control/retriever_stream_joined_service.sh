#!/bin/bash
set -euo pipefail
export E1_BASE=/weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004
export CUDA_VISIBLE_DEVICES=7 NVIDIA_TF32_OVERRIDE=0 E1_ENCODER_DEVICE=cpu
export PYTHONPATH="$E1_BASE/runtime/upstream/Search-R1:$E1_BASE/runtime:$E1_BASE/source"
export HF_HOME="$E1_BASE/hf-cache" HF_DATASETS_CACHE="$E1_BASE/hf-datasets" XDG_CACHE_HOME="$E1_BASE/cache" TMPDIR="$E1_BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=32 MKL_NUM_THREADS=16
"$E1_BASE/env/bin/python" "$E1_BASE/source/check_stream_admission.py"
exec "$E1_BASE/env/bin/python" "$E1_BASE/source/retriever_stream.py"

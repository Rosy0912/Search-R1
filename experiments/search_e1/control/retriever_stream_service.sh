#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?}
export CUDA_VISIBLE_DEVICES=7 NVIDIA_TF32_OVERRIDE=0 E1_ENCODER_DEVICE=cpu
export PYTHONPATH="$BASE/runtime/upstream/Search-R1:$BASE/runtime:$BASE/source"
export HF_HOME="$BASE/hf-cache" HF_DATASETS_CACHE="$BASE/hf-datasets" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=32 MKL_NUM_THREADS=16
"$BASE/env/bin/python" "$BASE/source/stream_admission.py"
exec "$BASE/env/bin/python" "$BASE/source/retriever_stream.py"

#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?};CONTROL="$BASE/source/experiments/search_e1/control"
mkdir -p "$BASE/tmp" "$BASE/cache" "$BASE/hf-datasets" "$BASE/receipts" "$BASE/logs"
export PYTHONPATH="$BASE/runtime/upstream/Search-R1:$BASE/runtime"
export HF_HOME="$BASE/hf-cache" HF_DATASETS_CACHE="$BASE/hf-datasets" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=16
while [[ ! -f "$BASE/receipts/assets.ready" ]]; do sleep 10; done
"$BASE/env/bin/python" "$CONTROL/gpu_admission.py" > "$BASE/receipts/gpu-admission.json"
printf '%s\n' "$$" > "$BASE/receipts/retriever.pid"
exec "$BASE/env/bin/python" "$CONTROL/retriever_gpu.py"

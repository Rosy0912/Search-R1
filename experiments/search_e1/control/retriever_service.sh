#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?};CONTROL="$BASE/source/experiments/search_e1/control"
export PYTHONPATH="$BASE/runtime/archive/upstream/Search-R1:$BASE/runtime/archive"
export HF_HOME="$BASE/hf-cache" HF_DATASETS_CACHE="$BASE/hf-datasets" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=16
printf '%s\n' "$$" > "$BASE/receipts/retriever.pid"
exec "$BASE/env/bin/python" "$CONTROL/retriever_cpu.py"

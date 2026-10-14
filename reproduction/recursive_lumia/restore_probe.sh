#!/bin/bash
set -euo pipefail
BASE="${RECURSIVE_LUMIA_BASE:?}"
SOURCE="${RECURSIVE_SOURCE_DIR:?}"
: "${RECURSIVE_SOURCE_COMMIT:?}"
: "${RECURSIVE_RESTORE_PROBE_DIRECTORY:?}"
: "${RECURSIVE_RESTORE_PROBE_INPUTS_SHA:?}"
RUNTIME=2dd059cc8bcf58ad378584cde8528482aea8a9ff
export PATH=/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/env/bin:$PATH
export PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0
export HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS
cd "$SOURCE"
python -m reproduction.recursive_lumia.verify_source "$SOURCE" "$BASE/receipts/source-$RECURSIVE_SOURCE_COMMIT.json"
python -m reproduction.recursive_lumia.verify_source "$BASE/sources/$RUNTIME" "$BASE/receipts/source-$RUNTIME.json"
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m reproduction.recursive_lumia.restore_probe run \
  --directory "$RECURSIVE_RESTORE_PROBE_DIRECTORY" --inputs-sha "$RECURSIVE_RESTORE_PROBE_INPUTS_SHA"

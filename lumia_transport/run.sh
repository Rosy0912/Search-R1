#!/bin/bash
set -euo pipefail
: "${RECURSIVE_SOURCE_DIR:?}" "${RECURSIVE_SOURCE_COMMIT:?}" "${RECURSIVE_LUMIA_BASE:?}"
: "${TRANSPORT_SOURCE:?}" "${TRANSPORT_COMMIT:?}" "${TRANSPORT_BINDING:?}" "${TRANSPORT_BINDING_SHA:?}"
export PATH=/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/env/bin:$PATH
export PYTHONPATH="$RECURSIVE_SOURCE_DIR:$TRANSPORT_SOURCE" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0
export VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="$RECURSIVE_LUMIA_BASE/cache/hf" XDG_CACHE_HOME="$RECURSIVE_LUMIA_BASE/cache" TRITON_CACHE_DIR="$RECURSIVE_LUMIA_BASE/cache/triton-$SLURM_JOB_ID"
export C_INCLUDE_PATH=/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/bootstrap-tools/python-dev/usr/include/python3.10:/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/bootstrap-tools/python-dev/usr/include
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS
cd "$RECURSIVE_SOURCE_DIR"
python -m reproduction.recursive_lumia.verify_source "$RECURSIVE_SOURCE_DIR" "$RECURSIVE_LUMIA_BASE/receipts/source-$RECURSIVE_SOURCE_COMMIT.json"
python -m reproduction.recursive_lumia.verify_source "$TRANSPORT_SOURCE" "$RECURSIVE_LUMIA_BASE/receipts/source-$TRANSPORT_COMMIT.json"
python -m lumia_transport.handoff "$TRANSPORT_BINDING" "$TRANSPORT_BINDING_SHA"
export RECURSIVE_ARM=C1-root-only
python -m reproduction.recursive_lumia.proxy "$RECURSIVE_LUMIA_BASE" > "$RECURSIVE_LUMIA_BASE/logs/proxy-$SLURM_JOB_ID.log" 2>&1 &
PROXY_PID=$!
trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
python -m lumia_transport.parent "$TRANSPORT_BINDING" "$TRANSPORT_BINDING_SHA"

#!/usr/bin/env bash
set -euo pipefail
umask 007
SOURCE="${1:?published immutable source required}"
shift
REUSED=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
BINARY="$REUSED/runtime/retriever-python310-candidate/bin/python3.10"
export QPROBE_SOURCE="$SOURCE" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SOURCE:$REUSED/runtime/gpu-faiss-alignment-candidate:$REUSED/source/env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$REUSED/runtime/retriever-python310-candidate/lib"
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# The short start/inspect path only locates the pinned private overlay. Each
# detached bound worker verifies every dependency byte before any CPU archive.
CPU_SITE="$($BINARY -c 'from reproduction.hpc3_qprobe.probe_cpu_runtime import root; print(root()/"site-packages")')"
export PYTHONPATH="$SOURCE:$CPU_SITE:$REUSED/runtime/gpu-faiss-alignment-candidate:$REUSED/source/env/lib/python3.10/site-packages"
cd "$SOURCE"
exec "$BINARY" -m reproduction.hpc3_qprobe.final_control "$@"

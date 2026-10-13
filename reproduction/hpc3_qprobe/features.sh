#!/usr/bin/env bash
set -euo pipefail
BASE="${1:?explicit remote base required}"
SOURCE="${2:?immutable committed source directory required}"
REUSED=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
PY="$REUSED/runtime/retriever-python310-candidate/bin/python3.10"
export PYTHONPATH="$SOURCE:$REUSED/source/env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$REUSED/runtime/retriever-python310-candidate/lib"
export HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$SOURCE"
exec "$PY" -m reproduction.hpc3_qprobe.precompute --base "$BASE"

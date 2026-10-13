#!/usr/bin/env bash
set -euo pipefail
BASE="${1:?explicit remote base required}"
SOURCE="${2:?immutable source required}"
export QPROBE_SOURCE="$SOURCE" PATH="$BASE/env/bin:$PATH"
export HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export TRITON_CACHE_DIR="$BASE/cache/debug-triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/debug-inductor"
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$SOURCE"
exec python -m reproduction.hpc3_qprobe.debug --base "$BASE"

#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?original remote qprobe base required}"
SOURCE="${2:?immutable published diagnostic source required}"
export QPROBE_SOURCE="$SOURCE" PYTHONUNBUFFERED=1
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$SOURCE"
exec "$BASE/env/bin/python" -m reproduction.hpc3_qprobe.gpu_inventory_diagnostic --output "$BASE/preparation/gpu-inventory717607/job-$SLURM_JOB_ID"

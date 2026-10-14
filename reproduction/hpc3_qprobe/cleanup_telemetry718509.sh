#!/usr/bin/env bash
set -euo pipefail
umask 007
SOURCE="${1:?immutable source required}"
BASE=/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718322-20261014
export QPROBE_SOURCE="$SOURCE" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SOURCE"
exec "$BASE/env/bin/python" -m reproduction.hpc3_qprobe.telemetry_failure718509 native-cleanup

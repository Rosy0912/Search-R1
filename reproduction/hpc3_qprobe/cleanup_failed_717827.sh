#!/usr/bin/env bash
set -euo pipefail
SOURCE="${1:?immutable published source required}"
export QPROBE_SOURCE="$SOURCE" PYTHONUNBUFFERED=1
exec /data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717765-20261014/env/bin/python -m reproduction.hpc3_qprobe.cleanup_failed_717827

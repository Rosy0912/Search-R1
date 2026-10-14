#!/usr/bin/env bash
set -euo pipefail
umask 007
BASE="${1:?new dedicated remote audit base required}"
COMMIT="${2:?published immutable audit source commit required}"
RECEIPT="${3:?actual completed200 paired source receipt required}"
TRANSPORT="${4:?source/archive/target complete full SHA transport proof required}"
SOURCE="/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/$COMMIT"
REUSED="/data/user/jxu957/experiments/cosearch-continue300-x203-20261008"
BINARY="$REUSED/runtime/retriever-python310-candidate/bin/python3.10"
export SEARCH_G1_BASE="$BASE" SEARCH_G1_PROFILE=common8 SEARCH_G1_TRAINING_WORLD_SIZE=4
export QPROBE_SOURCE="$SOURCE" SEARCH_G1_EXECUTION_PROFILE=native4_bf16_varlen_fp32_master
export PYTHONPATH="$SOURCE:$REUSED/runtime/gpu-faiss-alignment-candidate:$REUSED/source/env/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$REUSED/runtime/retriever-python310-candidate/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CONFIG_HOME="$BASE/cache/config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
CPU_SITE="$($BINARY -c 'from reproduction.hpc3_qprobe.probe_cpu_runtime import verify; print(verify()[0])')"
export PYTHONPATH="$SOURCE:$CPU_SITE:$REUSED/runtime/gpu-faiss-alignment-candidate:$REUSED/source/env/lib/python3.10/site-packages"
cd "$SOURCE"
exec "$BINARY" -m reproduction.hpc3_qprobe.final_restore_audit --base "$BASE" --commit "$COMMIT" --source-receipt "$RECEIPT" --transport-proof "$TRANSPORT"

#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009
OLDROOT=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009
SOURCE=0fe642bd0c71f0b5f9813cbbad7bdd94e0625244
TOKEN=${1:?unique token}
# CPU Slurm allocation; physical GPUs acquired by the canonical eight-UUID cohost leases.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
[[ "$TOKEN" =~ ^[a-f0-9]{32}$ && $(hostname) = h100-11 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_RESTART_COUNT:-0} = 0 && -n ${CUDA_VISIBLE_DEVICES:-} ]]
export SPT_NOENV=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$ROOT/env/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$ROOT/tools:$ROOT/retriever-gpu-lib:$ROOT/repo"
unset PYTHONOPTIMIZE LD_LIBRARY_PATH http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$ROOT/cache/hf" HF_DATASETS_CACHE="$ROOT/cache/datasets" XDG_CACHE_HOME="$ROOT/cache"
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
export TRITON_LIBCUDA_PATH="$ROOT/lib/driver" LIBRARY_PATH="$ROOT/lib/driver"
RUN="$ROOT/runs/eval150-final51713-$TOKEN-$SLURM_JOB_ID"
mkdir "$RUN"
export BAPO_RUN_ID="$RUN" SEARCH_R1_BASELINE_RUN_ROOT="$RUN"
export PYTHONPATH="$ROOT/tools/eval150:$PYTHONPATH"
cleanup() {
 status=$?; trap - EXIT
 "$ROOT/env/bin/python" -B "$ROOT/tools/resource_watch.py" --cleanup-run "$RUN" > "$RUN/cleanup-command.log" 2>&1; clean=$?
 printf '%s\n' "$status" > "$RUN/original-exit-code.txt"
 printf '%s\n' "$clean" > "$RUN/cleanup-exit-code.txt"
 exit "$status"
}
trap cleanup EXIT
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v11/bootstrap.py" --run "$RUN" --minimum-start-free-mib 30720 --reserve-free-mib 1024 -- bash "$ROOT/tools/eval150/inner.sh" > "$RUN/supervisor.log" 2>&1

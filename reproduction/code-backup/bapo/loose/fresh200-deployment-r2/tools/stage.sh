#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r2-20261006
PHASE=${1:?phase required}
case "$PHASE" in diagnostic-discovery128|diagnostic-discovery256|diagnostic-confirmation128|diagnostic-confirmation256|formal200) ;; *) exit 2;; esac
[[ $(hostname) = h100-5 ]]
[[ ${SLURM_RESTART_COUNT:-0} = 0 ]] || { echo "Restarted allocation forbidden" >&2; exit 3; }
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}"
export SPT_NOENV=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$ROOT/env/bin:$ROOT/tools:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$ROOT/repo"
unset PYTHONOPTIMIZE
ulimit -c 0
exec 9>"$ROOT/scheduler-head/worker-global.lock"
flock -n 9 || { echo 'Another worker owns execution' >&2; exit 3; }
python "$ROOT/tools/verify-launch.py" "$PHASE"
RUN=$(python "$ROOT/tools/launch_transaction.py" worker "$PHASE" "$SLURM_JOB_ID")
export BAPO_RUN_ID="$RUN" BAPO_FRESH200_RUN="$RUN"
export SEARCH_R1_ASSETS="$ROOT/assets-root" SEARCH_R1_REPO="$ROOT/repo"
COMMON=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005
export BAPO_FRESH200_DATA="$COMMON/common-data"
export BAPO_FRESH200_MODEL="$ROOT/assets/qwen2.5-3b-instruct-aa8e725"
export BAPO_FRESH200_DATA_MANIFEST_SHA256=0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24
export BAPO_FRESH200_RETRIEVER_URL=http://127.0.0.1:18083/retrieve
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline VLLM_NO_USAGE_STATS=1
export NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 RAY_USAGE_STATS_ENABLED=0
export TRITON_LIBCUDA_PATH="$ROOT/lib/driver" LIBRARY_PATH="$ROOT/lib/driver"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
mkdir -p "$RUN/runtime/tmp" "$RUN/runtime/ray" "$RUN/logs"
RUNTIME=/tmp/bapo-fresh200-$SLURM_JOB_ID
[[ ! -e "$RUNTIME" && ! -L "$RUNTIME" ]]
ln -s "$RUN/runtime" "$RUNTIME"
export SEARCH_R1_BASELINE_OUTPUT="$RUN/output" SEARCH_R1_BASELINE_RUN_ROOT="$RUNTIME"
export TMPDIR="$RUNTIME/tmp" RAY_TMPDIR="$RUNTIME/ray" HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache"
cp "$ROOT/source-expected.txt" "$RUN/source-commit.txt"
cleanup() {
 status=$?
 trap - EXIT
 cleanup_status=0
 python "$ROOT/tools/owned_processes.py" stop --run "$RUN" > "$RUN/own-process-cleanup.json" 2> "$RUN/own-process-cleanup.err" || cleanup_status=$?
 nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$RUN/gpu-exit.csv" 2> "$RUN/gpu-exit.err" || cleanup_status=1
 printf '%s\n' "$status" > "$RUN/original-exit-code.txt"
 printf '%s\n' "$cleanup_status" > "$RUN/cleanup-exit-code.txt"
 if [[ "$status" = 0 && "$cleanup_status" != 0 ]]; then status=$cleanup_status; fi
 if [[ -L "$RUNTIME" && $(readlink "$RUNTIME") = "$RUN/runtime" ]]; then rm "$RUNTIME"; fi
 exit "$status"
}
trap cleanup EXIT
cd "$ROOT/repo"
# The preflight leases pass inheritable UUID lock FDs to the guardian and all children.
python -m search_r1.baselines.preflight -- python "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash "$ROOT/tools/stage-gpu-inner.sh" "$PHASE" > "$RUN/phase.log" 2>&1

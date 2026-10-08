#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/probability-diagnostic-3307-v2-20261006
SOURCE_ROOT=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006
[[ $(hostname) = h100-5 && ${SLURM_RESTART_COUNT:-0} = 0 ]]
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}"
export PATH="$SOURCE_ROOT/env/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$ROOT/tools:$SOURCE_ROOT/repo"
export SPT_NOENV=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache" TRITON_CACHE_DIR="$ROOT/cache/triton"
unset PYTHONOPTIMIZE
ulimit -c 0
exec 9>"$ROOT/scheduler-head/worker-global.lock"
flock -n 9 || { echo 'Another diagnostic worker holds lock' >&2; exit 3; }
python -B "$ROOT/tools/verify-diagnostic.py" hf-zero-adam-replay
RUN=$(python -B "$ROOT/tools/launch_transaction.py" worker hf-zero-adam-replay "$SLURM_JOB_ID")
export BAPO_RUN_ID="$RUN" BAPO_DIAGNOSTIC_RECEIPT="$ROOT/scheduler-head/hf-zero-adam-replay-receipt.json"
# Wait only for the exact already-submitted receipt mirror; never submit/retry.
for ((i=0;i<60;i++)); do [[ -f "$BAPO_DIAGNOSTIC_RECEIPT" ]] && break;sleep 1;done
[[ -f "$BAPO_DIAGNOSTIC_RECEIPT" ]]
cp "$ROOT/source-expected.txt" "$RUN/source-commit.txt"
cleanup() {
 status=$?;trap - EXIT;cleanup_status=0
 python -B "$ROOT/tools/owned_processes.py" stop --run "$RUN" > "$RUN/own-process-cleanup.json" 2> "$RUN/own-process-cleanup.err" || cleanup_status=$?
 printf '%s\n' "$status" > "$RUN/original-exit-code.txt"
 printf '%s\n' "$cleanup_status" > "$RUN/cleanup-exit-code.txt"
 if [[ "$status" = 0 && "$cleanup_status" != 0 ]]; then status=$cleanup_status;fi
 exit "$status"
}
trap cleanup EXIT
cd "$SOURCE_ROOT/repo"
# The original fixed source preflight holds eight actual UUID locks; only
# cuda:0 computes. No retriever, Ray, trainer, optimizer or scheduler is started.
python -B -m search_r1.baselines.preflight -- python -B "$ROOT/tools/resource_watch.py" \
 --run "$RUN" -- bash "$ROOT/tools/phase.sh" > "$RUN/phase.log" 2>&1

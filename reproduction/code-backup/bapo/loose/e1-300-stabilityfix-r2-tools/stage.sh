#!/bin/bash
set -euo pipefail
ulimit -c 0
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005
PHASE=${1:?hardware|formal300}
case "$PHASE" in hardware|formal300) ;; *) exit 2;; esac
export PATH="$ROOT/env/bin:$ROOT/tools:$PATH"
export PYTHONPATH="$ROOT/repo"
export SEARCH_R1_ASSETS="$ROOT/assets-root" SEARCH_R1_REPO="$ROOT/repo"
export BAPO_E1_DATA="$ROOT/common-data" BAPO_E1_MODEL="$ROOT/assets/qwen2.5-3b-instruct-aa8e725"
export BAPO_E1_DATA_MANIFEST_SHA256=0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24
export BAPO_E1_RETRIEVER_URL=http://127.0.0.1:18083/retrieve
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 RAY_USAGE_STATS_ENABLED=0
export TRITON_LIBCUDA_PATH="$ROOT/lib/driver"
export LIBRARY_PATH="$ROOT/lib/driver${LIBRARY_PATH:+:$LIBRARY_PATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?Respect scheduler visibility}"
export SOURCE=$(cat "$ROOT/source-expected.txt")
cd "$ROOT/repo"
test "$(git rev-parse HEAD)" = "$SOURCE"
test -z "$(git status --porcelain)"
python "$ROOT/tools/verify_launch.py" "$PHASE"
python "$ROOT/tools/verify_shared_inputs.py"
RUN="$ROOT/runs/bapo-instruct300-${PHASE}-${SLURM_JOB_ID}"
mkdir "$RUN"
export BAPO_RUN_ID="$RUN" BAPO_E1_RUN="$RUN"
mkdir -p "$RUN/runtime/tmp" "$RUN/runtime/ray" "$RUN/logs"
RUNTIME="/tmp/bapo-e1-${SLURM_JOB_ID}"
ln -s "$RUN/runtime" "$RUNTIME"
test "$(readlink -f "$RUNTIME")" = "$RUN/runtime"
export SEARCH_R1_BASELINE_OUTPUT="$RUN/output" SEARCH_R1_BASELINE_RUN_ROOT="$RUNTIME"
export TMPDIR="$RUNTIME/tmp" HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache" RAY_TMPDIR="$RUNTIME/ray"
printf '%s\n' "$SOURCE" > "$RUN/source-commit.txt"
python "$ROOT/tools/cpu-tests.py" --origins-only > "$RUN/runtime-import-origins.log"
cp "$ROOT/logs/launch-approval.json" "$ROOT/logs/tools-manifest.json" "$ROOT/logs/cpu-gate.json" "$ROOT/logs/cpu-gate-source.txt" "$RUN/"
python -c 'import os; print("\n".join(k+"="+v for k,v in sorted(os.environ.items()) if k.startswith("SLURM_") or k=="CUDA_VISIBLE_DEVICES"))' > "$RUN/slurm-allocation.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv > "$RUN/gpus-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$RUN/processes-before.csv"
exec 9>"$ROOT/gpu-job.lock"
flock -n 9
cleanup(){
 BAPO_RUN_ID="$RUN" python "$ROOT/tools/owned_processes.py" stop --run "$RUN" > "$RUN/own-process-cleanup.json"
 if [ -L "$RUNTIME" ] && [ "$(readlink -f "$RUNTIME")" = "$RUN/runtime" ]; then unlink "$RUNTIME"; fi
 nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$RUN/gpus-exit.csv" || true
}
trap cleanup EXIT
if [ "$PHASE" = hardware ]; then
 test "$(cat "$ROOT/logs/cpu-gate-source.txt")" = "$SOURCE"
 bash "$ROOT/tools/monitor.sh" "$RUN" 0 0 &
 python -m search_r1.baselines.preflight -- python "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash "$ROOT/tools/hardware.sh" "$RUN" > "$RUN/hardware.log" 2>&1
else
 python "$ROOT/tools/verify_hardware_receipt.py" > "$RUN/hardware-receipt-check.log"
 python -m search_r1.baselines.preflight -- python "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash "$ROOT/tools/stage-gpu-inner.sh" "$PHASE" > "$RUN/phase.log" 2>&1
fi

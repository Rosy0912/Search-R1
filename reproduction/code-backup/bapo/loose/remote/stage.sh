#!/bin/bash
set -euo pipefail
ulimit -c 0
ROOT=/dataset/shaoshitong/search-r1-rl/bapo
PHASE=${1:?hardware|smoke2|accept5|formal500}
export PATH="$ROOT/env/bin:$ROOT/tools:$PATH"
export SEARCH_R1_ASSETS="$ROOT/assets-root" SEARCH_R1_REPO="$ROOT/repo"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export TRITON_LIBCUDA_PATH="$ROOT/lib/driver"
export LIBRARY_PATH="$ROOT/lib/driver${LIBRARY_PATH:+:$LIBRARY_PATH}"
export RAY_USAGE_STATS_ENABLED=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
: "${SLURM_JOB_ID:?Must run in dedicated Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Respect scheduler GPU visibility}"
SOURCE=b20f371972bb1bc41b735c6cd350b9a779d8820d
cd "$ROOT/repo"
test "$(git rev-parse HEAD)" = "$SOURCE"
test -z "$(git status --porcelain)"
test -f "$ROOT/logs/environment.ready"
RUN="$ROOT/runs/bapo-${PHASE}-${SLURM_JOB_ID}"
mkdir -p "$ROOT/runs"
mkdir "$RUN"
export BAPO_RUN_ID="$RUN"
mkdir -p "$RUN/runtime/tmp" "$RUN/runtime/ray"
# Ray Unix-domain sockets include a long session suffix; use an owned short alias.
RUNTIME="/tmp/bapo-${SLURM_JOB_ID}"
ln -s "$RUN/runtime" "$RUNTIME"
test "$(readlink -f "$RUNTIME")" = "$RUN/runtime"
export SEARCH_R1_BASELINE_OUTPUT="$RUN/output" SEARCH_R1_BASELINE_RUN_ROOT="$RUNTIME"
export TMPDIR="$RUNTIME/tmp" HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache" RAY_TMPDIR="$RUNTIME/ray"
printf '%s\n' "$SOURCE" > "$RUN/source-commit.txt"
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
if [ "$PHASE" = hardware ] || [ "$PHASE" = hardware-basic ]; then
  test -f "$ROOT/logs/cpu-gate.ready"
  test "$(cat "$ROOT/logs/cpu-gate-source.txt")" = "$SOURCE"
  bash "$ROOT/tools/monitor.sh" "$RUN" 0 0 &
  script=hardware.sh
  if [ "$PHASE" = hardware-basic ]; then script=hardware-basic.sh; fi
  python -m search_r1.baselines.preflight -- python "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash "$ROOT/tools/$script" "$RUN" > "$RUN/hardware.log" 2>&1
  exit 0
fi
case "$PHASE" in
 smoke2) steps=3; smoke=true; val_before=false;;
 accept5) steps=6; smoke=true; val_before=false;;
 formal500) steps=501; smoke=false; val_before=true;;
 *) exit 2;;
esac
python "$ROOT/tools/verify_hardware_receipt.py" > "$RUN/hardware-receipt-check.log"
cp "$ROOT/logs/hardware-acceptance.json" "$RUN/hardware-acceptance.json"
python -m search_r1.baselines.preflight -- python "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash "$ROOT/tools/stage-gpu-inner.sh" "$PHASE" > "$RUN/phase.log" 2>&1

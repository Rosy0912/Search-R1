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
SOURCE=a3e000de547e1260eadb1a20cb0a350d58c1133d
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
# Verify intended ports are free before starting only this allocation's services.
python - <<'PY'
import socket
for port in (18083,16383):
 s=socket.socket();s.bind(('127.0.0.1',port));s.close()
PY
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python reproduction/h20/retriever_local.py \
 --index_path "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" \
 --corpus_path "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" \
 --retriever_model "$SEARCH_R1_ASSETS/assets/e5-base-v2" \
 --device cpu --host 127.0.0.1 --port 18083 --threads 8 > "$RUN/retriever.log" 2>&1 &
retriever_pid=$!
ray_pid=''
monitor_pid=''
(bash "$ROOT/tools/monitor.sh" "$RUN" "$retriever_pid" 0) &
monitor_pid=$!
ready=false
for ((i=0;i<360;i++)); do
 kill -0 "$retriever_pid"
 if curl -fsS --max-time 3 http://127.0.0.1:18083/openapi.json >/dev/null; then ready=true;break;fi
 sleep 5
done
test "$ready" = true
curl -fsS --max-time 60 -H 'Content-Type: application/json' --data '{"queries":["What is the capital of France?"],"topk":50,"return_scores":true}' http://127.0.0.1:18083/retrieve > "$RUN/retriever-health.json"
python "$ROOT/tools/ray_cli_compat.py" start --head --block --port=16383 --min-worker-port=20000 --max-worker-port=29999 --num-gpus=8 --num-cpus=24 --object-store-memory=8589934592 --include-dashboard=false --temp-dir="$RUNTIME/ray" > "$RUN/ray-head.log" 2>&1 &
ray_pid=$!
export RAY_ADDRESS=127.0.0.1:16383
for ((i=0;i<60;i++)); do
 kill -0 "$ray_pid"
 if python -c 'import socket; socket.create_connection(("127.0.0.1",16383),timeout=1).close()' 2>/dev/null; then break;fi
 sleep 2
done
python -c 'import socket; socket.create_connection(("127.0.0.1",16383),timeout=1).close()'
args=(actor_rollout_ref.actor.ppo_micro_batch_size=8 retriever.url=http://127.0.0.1:18083/retrieve
 "trainer.total_training_steps=$steps" "trainer.smoke_only=$smoke" "trainer.val_before_train=$val_before"
 "trainer.experiment_name=bapo-$PHASE-$SLURM_JOB_ID" "baselines.algorithm_source_commit=$SOURCE")
bash reproduction/baselines/run.sh bapo "${args[@]}" --cfg job --resolve > "$RUN/resolved-config.yaml"
/usr/bin/time -v python "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash reproduction/baselines/run.sh bapo "${args[@]}" > "$RUN/train.log" 2> "$RUN/driver-time.log"
touch "$RUN/TRAIN_RETURNED_SUCCESSFULLY"

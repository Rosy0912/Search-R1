#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
STAGE=${1:?two|five|formal}
case "$STAGE" in
 two) STEPS=3; SMOKE=true; VALIDATE=false ;;
 five) STEPS=6; SMOKE=true; VALIDATE=false ;;
 formal) STEPS=501; SMOKE=false; VALIDATE=true ;;
 *) exit 2 ;;
esac
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NODELIST:-}" = h100-9
source "$ROOT/env/bin/activate"
cd "$ROOT/repo"
# The evidence review must happen before any training allocation is dispatched.
python - "$ROOT/logs/retrieval-acceptance.json" "$STAGE" <<'CHECK'
import hashlib,json,pathlib,sys,os
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
if sys.argv[2]!='two':
 previous='two' if sys.argv[2]=='five' else 'five'
 a=json.loads(pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/'+previous+'-h100-native-stage-acceptance.json').read_text())
 assert a['status']=='REAL_TREE_STAGE_ACCEPTED' and a['stage']==previous and a['actual_outer_updates']=={'two':2,'five':5}[previous]
 for f,h in a['evidence_file_sha256'].items():assert hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h, 'Accepted stage evidence changed'
inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
assert len(selected)==8 and {x['uuid'] for x in selected}==set(json.load(open('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json')))
CHECK
git diff --quiet f4f457df55fbfe99691d466d5c6f1d3fd7175c65 -- search_r1 verl
RUN="$ROOT/runs/tree-$STAGE-$SLURM_JOB_ID"
mkdir "$RUN"
mkdir "$RUN/runtime"
# Short socket spelling, persistent target; no reuse of an existing alias.
SHORT_RUNTIME="/tmp/sst-tree-$SLURM_JOB_ID"
test ! -e "$SHORT_RUNTIME" && test ! -L "$SHORT_RUNTIME"
ln -sT "$RUN/runtime" "$SHORT_RUNTIME"
BACKEND=h100-native
RETRIEVAL_URL=http://10.32.0.11:18082/retrieve
ACCEPTANCE_FILE=${TREE_H100_RETRIEVAL_ACCEPTANCE_FILE:?}
ACCEPTANCE_SHA=${TREE_H100_RETRIEVAL_ACCEPTANCE_SHA256:?}
export SEARCH_R1_TREE_AUDIT_DIR="$RUN/output/tree-audit"
export SEARCH_R1_ASSETS="$ROOT/assets-root" SEARCH_R1_BASELINE_OUTPUT="$RUN/output" SEARCH_R1_BASELINE_RUN_ROOT="$SHORT_RUNTIME"
export C_INCLUDE_PATH="$ROOT/python-dev/usr/include/python3.10:$ROOT/python-dev/usr/include"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 VLLM_NO_USAGE_STATS=1 VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline WANDB_DIR="$RUN" PYTHONPATH="$ROOT/repo"
unset WANDB_API_KEY
export NO_PROXY=127.0.0.1,localhost,10.32.0.11
export no_proxy="$NO_PROXY"
mkdir -p "$RUN/runtime/ray" "$RUN/runtime/tmp" "$RUN/cache"
export TMPDIR="$RUN/runtime/tmp" HF_DATASETS_CACHE="$ROOT/retriever-cache/datasets"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache/torchinductor" TRITON_CACHE_DIR="$RUN/cache/triton" CUDA_CACHE_PATH="$RUN/cache/cuda" XDG_CACHE_HOME="$RUN/cache"
exec > >(tee "$RUN/stage.log") 2>&1
start=$(date +%s)
RETRIEVER_PID=''
GUARD_PID=''
CPU_SAMPLER_PID=''
finish() {
 code=$?
 trap - EXIT
 signal_epoch=$(date +%s)
 printf '{"job_id":%s,"stage":"%s","start_epoch":%s,"termination_epoch":%s,"allocated_gpus":8,"exit_code":%s,"scope":"stage body until finish entry; cleanup/end may be cut short by Slurm; allocation end must be recorded independently"}\n' "$SLURM_JOB_ID" "$STAGE" "$start" "$signal_epoch" "$code" > "$RUN/termination-cost.json"
 if [ -n "$RETRIEVER_PID" ]; then
  kill -TERM "$RETRIEVER_PID" 2>/dev/null || true
  for n in $(seq 1 15); do kill -0 "$RETRIEVER_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$RETRIEVER_PID" 2>/dev/null; then kill -KILL "$RETRIEVER_PID" 2>/dev/null || true; fi
  wait "$RETRIEVER_PID" 2>/dev/null || true
 fi
 if [ -n "$CPU_SAMPLER_PID" ]; then kill -TERM "$CPU_SAMPLER_PID" 2>/dev/null || true; wait "$CPU_SAMPLER_PID" 2>/dev/null || true; fi
 if [ -n "$GUARD_PID" ]; then kill -TERM "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true; fi
 end=$(date +%s)
 printf '{"job_id":%s,"stage":"%s","start_epoch":%s,"end_epoch":%s,"allocated_gpus":8,"exit_code":%s,"scope":"stage body, excludes scheduler prolog/epilog"}\n' "$SLURM_JOB_ID" "$STAGE" "$start" "$end" "$code" > "$RUN/cost.json"
 # Slurm Proctrack/cgroup removes this job's Ray descendants at allocation end.
 exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python - "$BACKEND" <<'PORTS'
import socket,sys
ports=[16382]
for port in ports:
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
PORTS
python "$ROOT/tools/resource_guard.py" --output "$RUN/resource-samples.jsonl" &
GUARD_PID=$!
check_native_service() {
 python "$ROOT/tools/check_h100_native_retriever.py" --receipt "$ACCEPTANCE_FILE" --sha256 "$ACCEPTANCE_SHA" --stage "$STAGE" --required-seconds "${1:-0}"
}
check_native_service "${TREE_STAGE_MAX_SECONDS:?Bounded Slurm stage lifetime required}" > "$RUN/h100-native-bound-provenance.json"
python "$ROOT/tools/check_actor_compile.py" --output "$RUN/actor-compile-probe.json"
ray start --head --port=16382 --min-worker-port=22082 --max-worker-port=23081 --num-gpus=8 --num-cpus=24 --object-store-memory=8589934592 --include-dashboard=false --temp-dir="$SHORT_RUNTIME/ray"
export RAY_ADDRESS=127.0.0.1:16382
args=(actor_rollout_ref.actor.ppo_micro_batch_size=8 retriever.url="$RETRIEVAL_URL" baselines.holdout.output_dir="$ROOT/common-data" trainer.total_training_steps="$STEPS" trainer.smoke_only="$SMOKE" trainer.val_before_train="$VALIDATE" trainer.logger='[console,wandb]')
bash reproduction/baselines/run.sh tree_grpo "${args[@]}" --cfg job --resolve > "$RUN/resolved.yaml"
if [ -n "$RETRIEVER_PID" ]; then kill -0 "$RETRIEVER_PID"; fi
kill -0 "$GUARD_PID"
bash reproduction/baselines/run.sh tree_grpo "${args[@]}" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
 service_ok=true
 service_rc=0
 timeout --signal=TERM --kill-after=5 30 bash -c 'python "$1" --receipt "$2" --sha256 "$3" --stage "$4" --required-seconds 0' _ "$ROOT/tools/check_h100_native_retriever.py" "$ACCEPTANCE_FILE" "$ACCEPTANCE_SHA" "$STAGE" >> "$RUN/h100-native-health-check.log" 2>&1 || { service_rc=$?; service_ok=false; }
 if [ "$service_ok" != true ] || ! kill -0 "$GUARD_PID" 2>/dev/null; then
  printf '{"epoch":%s,"job_id":%s,"start_epoch":%s,"allocated_gpus":8,"checker_exit_code":%s,"service_ok":%s,"scope":"own caller failure before scancel; not allocation-end cost"}\n' "$(date +%s)" "$SLURM_JOB_ID" "$start" "$service_rc" "$service_ok" > "$RUN/service-stop.json"
  echo REQUIRED_SERVICE_DIED_STOP_OWN_JOB
  scancel --signal=TERM "$SLURM_JOB_ID" || true
  # Fallback ends only the own driver; allocation end removes own Ray cgroup.
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  exit 1
 fi
 sleep 5
done
wait "$TRAIN_PID"
printf '%s\n' "$RUN" > "$ROOT/logs/$STAGE-stage-completed.run"

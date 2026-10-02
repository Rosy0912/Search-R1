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
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text());assert r['status']=='ACCEPTED_AFTER_REAL_DIFFERENCE_REVIEW'
for f,h in r['evidence_file_sha256'].items():assert hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h, 'Reviewed evidence changed'
if sys.argv[2]!='two':
 previous='two' if sys.argv[2]=='five' else 'five'
 a=json.loads(pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/logs/'+previous+'-stage-acceptance.json').read_text())
 assert a['status']=='REAL_TREE_STAGE_ACCEPTED' and a['stage']==previous
inventory,processes=gpu_snapshot();selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
assert len(selected)==8 and {x['uuid'] for x in selected}==set(json.load(open('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json')))
CHECK
git diff --quiet aed28e176da946dc6cff73cfb23fac5cf2a879a8 -- search_r1 verl
RUN="$ROOT/runs/tree-$STAGE-$SLURM_JOB_ID"
mkdir "$RUN"
mkdir "$RUN/runtime"
# Short socket spelling, persistent target; no reuse of an existing alias.
SHORT_RUNTIME="/tmp/sst-tree-$SLURM_JOB_ID"
test ! -e "$SHORT_RUNTIME" && test ! -L "$SHORT_RUNTIME"
ln -sT "$RUN/runtime" "$SHORT_RUNTIME"
export SEARCH_R1_ASSETS="$ROOT/assets-root" SEARCH_R1_BASELINE_OUTPUT="$RUN/output" SEARCH_R1_BASELINE_RUN_ROOT="$SHORT_RUNTIME"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 VLLM_NO_USAGE_STATS=1 VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR="$RUN" PYTHONPATH="$ROOT/repo"
unset WANDB_API_KEY
mkdir -p "$RUN/runtime/ray" "$RUN/runtime/tmp" "$RUN/cache"
export TMPDIR="$RUN/runtime/tmp" HF_DATASETS_CACHE="$ROOT/retriever-cache/datasets"
exec > >(tee "$RUN/stage.log") 2>&1
start=$(date +%s)
RETRIEVER_PID=''
GUARD_PID=''
finish() {
 code=$?
 trap - EXIT
 if [ -n "$RETRIEVER_PID" ]; then kill -TERM "$RETRIEVER_PID" 2>/dev/null || true; wait "$RETRIEVER_PID" 2>/dev/null || true; fi
 if [ -n "$GUARD_PID" ]; then kill -TERM "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true; fi
 end=$(date +%s)
 printf '{"job_id":%s,"stage":"%s","start_epoch":%s,"end_epoch":%s,"allocated_gpus":8,"exit_code":%s,"scope":"stage body, excludes scheduler prolog/epilog"}\n' "$SLURM_JOB_ID" "$STAGE" "$start" "$end" "$code" > "$RUN/cost.json"
 # Slurm Proctrack/cgroup removes this job's Ray descendants at allocation end.
 exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python - <<'PORTS'
import socket
for port in [18082,16382]:
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
PORTS
python "$ROOT/tools/resource_guard.py" --output "$RUN/resource-samples.jsonl" &
GUARD_PID=$!
# No GPU encoder/index and no GPU FAISS overlay in this process.
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python search_r1/search/retrieval_server.py --device cpu --host 127.0.0.1 --port 18082 --threads 8 --index_path "$ROOT/assets-root/assets/wiki/e5_Flat.index" --corpus_path "$ROOT/assets-root/assets/wiki/wiki-18.jsonl" --retriever_model "$ROOT/assets-root/assets/e5-base-v2" --topk 3 > "$RUN/retriever.log" 2>&1 &
RETRIEVER_PID=$!
for attempt in $(seq 1 240); do
 kill -0 "$RETRIEVER_PID" && kill -0 "$GUARD_PID"
 if curl -fsS --max-time 10 -o "$RUN/retriever-health.json" http://127.0.0.1:18082/openapi.json; then break; fi
 sleep 5
done
test -s "$RUN/retriever-health.json"
ray start --head --port=16382 --min-worker-port=22082 --max-worker-port=23081 --num-gpus=8 --num-cpus=24 --object-store-memory=8589934592 --include-dashboard=false --temp-dir="$SHORT_RUNTIME/ray"
export RAY_ADDRESS=127.0.0.1:16382
args=(actor_rollout_ref.actor.ppo_micro_batch_size=8 retriever.url=http://127.0.0.1:18082/retrieve baselines.holdout.output_dir="$ROOT/common-data" trainer.total_training_steps="$STEPS" trainer.smoke_only="$SMOKE" trainer.val_before_train="$VALIDATE" trainer.logger='[console,wandb]')
bash reproduction/baselines/run.sh tree_grpo "${args[@]}" --cfg job --resolve > "$RUN/resolved.yaml"
kill -0 "$RETRIEVER_PID" && kill -0 "$GUARD_PID"
bash reproduction/baselines/run.sh tree_grpo "${args[@]}" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
 if ! kill -0 "$RETRIEVER_PID" 2>/dev/null || ! kill -0 "$GUARD_PID" 2>/dev/null; then
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

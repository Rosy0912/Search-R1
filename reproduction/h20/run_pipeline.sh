#!/bin/bash
set -euo pipefail
ulimit -c 0
BASE=/data/search-r1
export SEARCH_R1_RUN_STARTED=$(date +%s)
export NCCL_NVLS_ENABLE=0
export PATH="$BASE/env/bin:$PATH"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=32
mkdir -p "$BASE/ray" "$BASE/checkpoints"
stage(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" | tee "$BASE/logs/pipeline-stage.txt"; }
trap 'stage "FAILED at line $LINENO; see pipeline.log"' ERR
stage WAITING_ENVIRONMENT_AND_ASSETS
for ((i=0;i<720;i++)); do
  if [ -f "$BASE/logs/environment.ready" ] && [ -f "$BASE/logs/assets-status.json" ]; then break; fi
  sleep 10
done
test -f "$BASE/logs/environment.ready"
python "$BASE/verify_assets.py" > "$BASE/logs/asset-verification.json"
exec 9>"$BASE/gpu-job.lock"
flock -w 600 9
export SEARCH_R1_LOCK_HELD=1
stage ENVIRONMENT_GPU_PREFLIGHT
python "$BASE/gpu_guard.py"
PREFLIGHT_PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$PREFLIGHT_PORT" --nproc_per_node=8 "$BASE/cuda_preflight.py" > "$BASE/logs/env-cuda-preflight.log" 2>&1
stage REAL_MODEL_INFERENCE_PREFLIGHT
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_NO_USAGE_STATS=1 python -X faulthandler "$BASE/model_smoke.py" > "$BASE/logs/model-smoke-fixed.log" 2>&1
stage PREPARING_DATA_AND_INDEX
python "$BASE/prepare_data.py"
python "$BASE/prepare_paper_data.py"
if [ ! -f "$BASE/assets/wiki/e5_Flat.index" ]; then
  cat "$BASE/assets/wiki/part_aa" "$BASE/assets/wiki/part_ab" > "$BASE/assets/wiki/e5_Flat.index.partial"
  mv "$BASE/assets/wiki/e5_Flat.index.partial" "$BASE/assets/wiki/e5_Flat.index"
fi
if [ ! -f "$BASE/assets/wiki/wiki-18.jsonl" ]; then
  gzip -dc "$BASE/assets/wiki/wiki-18.jsonl.gz" > "$BASE/assets/wiki/wiki-18.jsonl.partial"
  mv "$BASE/assets/wiki/wiki-18.jsonl.partial" "$BASE/assets/wiki/wiki-18.jsonl"
fi
stage STARTING_RETRIEVER
python "$BASE/gpu_guard.py"
python - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1',8000))
s.close()
PY
CUDA_VISIBLE_DEVICES=0 python "$BASE/retriever_local.py" \
  --index_path "$BASE/assets/wiki/e5_Flat.index" \
  --corpus_path "$BASE/assets/wiki/wiki-18.jsonl" \
  --topk 3 --retriever_name e5 --retriever_model "$BASE/assets/e5-base-v2" \
  > "$BASE/logs/retriever.log" 2>&1 &
RETRIEVER_PID=$!
# Stop only the recorded retriever identity, using Linux pidfd.
trap 'python "$BASE/process_identity.py" stop' EXIT
python "$BASE/process_identity.py" record "$RETRIEVER_PID"
READY=false
for ((i=0;i<360;i++)); do
  kill -0 "$RETRIEVER_PID"
  if curl -fsS --max-time 5 http://127.0.0.1:8000/openapi.json > /dev/null; then READY=true; break; fi
  sleep 10
done
test "$READY" = true
curl -fsS --max-time 120 -H 'Content-Type: application/json' \
  --data '{"queries":["capital of France"],"topk":3,"return_scores":true}' \
  http://127.0.0.1:8000/retrieve > "$BASE/logs/retrieval-test.json"
python -c 'import json;d=json.load(open("/data/search-r1/logs/retrieval-test.json"));assert len(d["result"][0])==3'
stage SMOKE_TRAINING
python "$BASE/gpu_guard.py"
bash "$BASE/train_smoke.sh"
python "$BASE/verify_checkpoint.py" > "$BASE/logs/checkpoint-update.json"
stage SMOKE_ACCEPTED
if [ "${SEARCH_R1_FULL_RUN:-0}" != 1 ]; then exit 0; fi
stage PAPER_GRPO_TRAINING
# Allow workers from the completed smoke driver to exit; never kill foreign jobs.
for ((i=0;i<30;i++)); do
  if python "$BASE/gpu_guard.py"; then break; fi
  sleep 2
done
python "$BASE/gpu_guard.py"
bash "$BASE/train_paper_grpo.sh"
python "$BASE/verify_full_run.py" > "$BASE/logs/full-run-acceptance.json"
stage FINISHED_TRAINING_AND_VALIDATION

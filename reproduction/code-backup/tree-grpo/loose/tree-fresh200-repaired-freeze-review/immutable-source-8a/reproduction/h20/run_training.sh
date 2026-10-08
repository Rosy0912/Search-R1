#!/bin/bash
# Restart training after the previously completed environment/asset/GPU preflights.
set -euo pipefail
BASE=/data/search-r1
export PATH="$BASE/env/bin:$PATH"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=32 NCCL_NVLS_ENABLE=0
export SEARCH_R1_RUN_STARTED=$(date +%s)
stage(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" | tee "$BASE/logs/pipeline-stage.txt"; }
trap 'stage "FAILED training supervisor at line $LINENO"' ERR
exec 9>"$BASE/gpu-job.lock"
flock -n 9
export SEARCH_R1_LOCK_HELD=1
python "$BASE/gpu_guard.py"
test -f "$BASE/logs/environment.ready"
python - <<'PY'
import json
from pathlib import Path
b=Path('/data/search-r1')
a=json.loads((b/'logs/asset-verification.json').read_text())
assert a['asset_count']==30 and a['content_hashes_verified']==30 and not a['failed']
assert 'EIGHT_GPU_NCCL_BF16_OK' in (b/'logs/env-cuda-preflight.log').read_text()
assert 'REAL_MODEL_VLLM_CUDA_OK' in (b/'logs/model-smoke-fixed.log').read_text()
g=json.loads((b/'logs/gpu-retrieval-preflight.json').read_text())
assert g['faiss_version']=='1.8.0' and g['top3_id_overlap']>=0.95 and g['gpu_seconds_512_queries']<30
PY
stage PREPARING_DIVERSE_64
python "$BASE/prepare_paper_data.py"
stage STARTING_RETRIEVER
python - <<'PY'
import socket
s=socket.socket();s.bind(('127.0.0.1',8000));s.close()
PY
PYTHONPATH="$BASE/retriever-gpu-lib:$BASE/repo" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "$BASE/retriever_local.py" \
  --index_path "$BASE/assets/wiki/e5_Flat.index" --corpus_path "$BASE/assets/wiki/wiki-18.jsonl" \
  --topk 3 --retriever_name e5 --retriever_model "$BASE/assets/e5-base-v2" --faiss_gpu \
  > "$BASE/logs/retriever.log" 2>&1 &
RETRIEVER_PID=$!
trap 'python "$BASE/process_identity.py" stop' EXIT
python "$BASE/process_identity.py" record "$RETRIEVER_PID"
READY=false
for ((i=0;i<360;i++)); do
  kill -0 "$RETRIEVER_PID"
  if curl -fsS --max-time 5 http://127.0.0.1:8000/openapi.json > /dev/null; then READY=true; break; fi
  sleep 5
done
test "$READY" = true
curl -fsS --max-time 120 -H 'Content-Type: application/json' \
  --data '{"queries":["capital of France"],"topk":3,"return_scores":true}' \
  http://127.0.0.1:8000/retrieve > "$BASE/logs/retrieval-test.json"
python -c 'import json; d=json.load(open("/data/search-r1/logs/retrieval-test.json")); assert len(d["result"][0])==3'
python "$BASE/gpu_guard.py"
printf '%s\n' "$SEARCH_R1_RUN_STARTED" > "$BASE/logs/training-run-started.txt"
stage TRAINING_64_VALIDATION_500_UPDATES
bash "$BASE/train_paper_grpo.sh"
python "$BASE/verify_full_run.py" > "$BASE/logs/full-run-acceptance.json"
stage FINISHED_TRAINING_AND_VALIDATION

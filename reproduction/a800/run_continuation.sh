#!/bin/bash
set -euo pipefail
BASE=/data/search-r1
unset PYTHONPATH PYTHONHOME RAY_ADDRESS
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
mkdir -p "$BASE/ray-a800" "$BASE/tmp" "$BASE/cache"
exec 9>"$BASE/gpu-job.lock"
flock -n 9
export SEARCH_R1_LOCK_HELD=1
stage(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" | tee "$BASE/logs/a800-pipeline-stage.txt"; }
trap 'stage "FAILED supervisor at line $LINENO"' ERR
cd "$BASE/repo"
stage WAITING_FOR_TRANSFERRED_CHECKPOINT
READY=false
for ((i=0;i<360;i++)); do
    if [[ -f "$BASE/transfer/checkpoint-100.tar.zst" && -f "$BASE/checkpoints/migration-source/global_step_100/model.safetensors.index.json" ]]; then READY=true; break; fi
    sleep 5
done
test "$READY" = true
python reproduction/a800/gpu_idle.py 0 1 2 3 5 6
test -f "$BASE/logs/environment.ready"
grep -q FOUR_GPU_NCCL_BF16_OK "$BASE/logs/a800-cuda-preflight.log"
grep -q A800_CURRENT_PREFLIGHT_COMPLETE "$BASE/logs/a800-current-preflight.log"
python - <<'PY'
import json
from pathlib import Path
b=Path('/data/search-r1')
assert json.loads((b/'logs/assets-status.json').read_text())['complete'] is True
g=json.loads((b/'logs/a800-gpu-retrieval-preflight-2gpu.json').read_text())
assert g['top3_id_overlap']>=.95 and g['query_count']==14 and g['faiss_version']=='1.8.0'
assert 'physical devices5,6' in g['mode']
import pandas as pd
root=b/'data/paper-grpo'
audit=json.loads((root/'dataset-audit.json').read_text())
assert sum(audit['eval_counts'].values())==51713 and audit['eval_counts']['nq']==3610
for name,count in [('train.parquet',169615),('validation-64.parquet',64),('test.parquet',51713)]:
    assert len(pd.read_parquet(root/name,columns=['data_source']))==count
assert (b/'retriever_local.py').read_bytes()==(b/'repo/reproduction/a800/retriever_local.py').read_bytes()
PY
stage VERIFYING_CONTINUATION_SOURCE
python reproduction/a800/prepare_continuation.py
export SEARCH_R1_SEGMENT_LOG=$(python -c 'import json; print(json.load(open("/data/search-r1/logs/a800-continuation-manifest.json"))["train_log"])')
RUN_ID=$(python -c 'import json; print(json.load(open("/data/search-r1/logs/a800-continuation-manifest.json"))["run_id"])')
cleanup(){
    touch "$BASE/logs/a800-monitor-$RUN_ID.stop"
    python "$BASE/process_identity.py" stop
}
trap cleanup EXIT
python reproduction/a800/gpu_monitor.py > "$BASE/logs/a800-monitor-error.log" 2>&1 &
python - <<'PY'
import socket
with socket.socket() as s:s.bind(('127.0.0.1',18000))
PY
stage STARTING_LOCAL_A800_RETRIEVER
python reproduction/a800/gpu_idle.py 5 6
PYTHONPATH="$BASE/retriever-gpu-lib:$BASE/repo" CUDA_VISIBLE_DEVICES=5,6 python "$BASE/retriever_local.py" \
  --index_path "$BASE/assets/wiki/e5_Flat.index" --corpus_path "$BASE/assets/wiki/wiki-18.jsonl" \
  --topk 3 --retriever_name e5 --retriever_model "$BASE/assets/e5-base-v2" --faiss_gpu \
  > "$BASE/logs/a800-retriever.log" 2>&1 &
RETRIEVER_PID=$!
python "$BASE/process_identity.py" record "$RETRIEVER_PID"
READY=false
for ((i=0;i<360;i++)); do
    kill -0 "$RETRIEVER_PID"
    if curl -fsS --max-time 5 http://127.0.0.1:18000/openapi.json > /dev/null; then READY=true; break; fi
    sleep 5
done
test "$READY" = true
curl -fsS --max-time 120 -H 'Content-Type: application/json' \
  --data '{"queries":["capital of France"],"topk":3,"return_scores":true}' \
  http://127.0.0.1:18000/retrieve > "$BASE/logs/a800-retrieval-http-test.json"
python -c 'import json; d=json.load(open("/data/search-r1/logs/a800-retrieval-http-test.json")); assert len(d["result"][0])==3'
python reproduction/a800/gpu_idle.py 0 1 2 3
stage CONTINUATION_101_TO_500
python reproduction/a800/acceptance_watch.py > "$BASE/logs/a800-five-update-watch.log" 2>&1 &
bash reproduction/a800/train_continuation.sh
stage TRAINER_FINISHED_PENDING_FINAL_ACCEPTANCE
python reproduction/a800/verify_full_continuation.py
stage FINISHED_TRAINING_AND_VALIDATION

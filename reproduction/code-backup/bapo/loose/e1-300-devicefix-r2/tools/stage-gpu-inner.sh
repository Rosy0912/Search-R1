#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-devicefix-r2-20261004
PHASE=$1
RUN=$BAPO_RUN_ID
RUNTIME=$SEARCH_R1_BASELINE_RUN_ROOT
SOURCE=$(cat "$RUN/source-commit.txt")
cd "$ROOT/repo"
python "$ROOT/tools/verify_shared_inputs.py"
# Verify intended ports are free before starting only this allocation's services.
python - <<'PY'
import socket
for port in (18083,16383):
 s=socket.socket();s.bind(('127.0.0.1',port));s.close()
PY
PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python reproduction/h20/retriever_local.py \
 --index_path "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" \
 --corpus_path "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" \
 --retriever_model "$SEARCH_R1_ASSETS/assets/e5-base-v2" \
 --device cuda --faiss_gpu --host 127.0.0.1 --port 18083 --threads 8 > "$RUN/retriever.log" 2>&1 &
retriever_pid=$!
printf '%s\n' 'same-node CUDA E5 FP16 + GPU FAISS half-shards; loopback127.0.0.1:18083; stage64; actorMicro16/logprobMicro32; Instruct; sequence_mean; 300outer' > "$RUN/runtime-profile.txt"
python - <<'PYPROOF' > "$RUN/retrieval-assets-locality.json"
import os,json,pathlib,subprocess
root=pathlib.Path(os.environ['SEARCH_R1_ASSETS']).resolve()
paths=[root/'assets/wiki/e5_Flat.index',root/'assets/wiki/wiki-18.jsonl',root/'assets/e5-base-v2']
records=[]
for p in paths:
 real=p.resolve(); allowed=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets').resolve(); assert real==allowed or allowed in real.parents
 fs=subprocess.check_output(['findmnt','-T',str(real),'-n','-o','SOURCE,FSTYPE,TARGET'],text=True).strip()
 assert fs.split()[1] in ('ext4','xfs'),fs
 records.append(dict(path=str(real),mount=fs))
print(json.dumps(dict(host=os.uname().nodename,paths=records,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],proxy_env=[k for k in os.environ if k.lower() in ('http_proxy','https_proxy','all_proxy')]),indent=2))
PYPROOF
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
args=(--config-name ppo_bapo_e1_300 "baselines.algorithm_source_commit=$SOURCE")
python -m search_r1.baselines.main "${args[@]}" --cfg job --resolve > "$RUN/resolved-config.yaml"
/usr/bin/time -v python -m search_r1.baselines.main "${args[@]}" > "$RUN/train.log" 2> "$RUN/driver-time.log"
touch "$RUN/TRAIN_RETURNED_SUCCESSFULLY"

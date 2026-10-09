#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v9-20261009
RUN=${BAPO_RUN_ID:?}
cd "$ROOT/repo"
# The worker has exclusively claimed RUN; do not invoke source run.sh again.
eval "$("$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_admission.py" environment --run "$RUN")"
export SEARCH_R1_RETRIEVER_IDENTITY=1
"$ROOT/env/bin/python" -B - <<'PY'
import socket
for port in (18083,16383):
 s=socket.socket();s.bind(('127.0.0.1',port));s.close()
PY
"$ROOT/env/bin/python" -B reproduction/h20/retriever_local.py --index_path "/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/wiki/e5_Flat.index" --corpus_path "/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/wiki/wiki-18.jsonl" --retriever_model "/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/e5-base-v2" --device cuda --faiss_gpu --host 127.0.0.1 --port 18083 --threads 8 > "$RUN/retriever.log" 2>&1 &
retriever_pid=$!
export BAPO_LOCAL_RETRIEVER_PID="$retriever_pid" BAPO_RETRIEVER_PARENT_PID="$$"
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_service_proof.py" retriever-freeze
ready=false
for ((i=0;i<360;i++)); do
 kill -0 "$retriever_pid"
 if curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:18083/identity > "$RUN/identity-ready.tmp"; then ready=true;break;fi
 sleep 5
done
[[ "$ready" = true ]]
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_service_proof.py" retriever-check
# Verify loopback belongs to this newly launched marker/job, before Ray allocation.
"$ROOT/env/bin/python" -B - <<'PY'
import json,os
from pathlib import Path
import requests
s=requests.Session();s.trust_env=False
r=s.get('http://127.0.0.1:18083/identity',timeout=(5,30));r.raise_for_status();identity=r.json()
pid=int(os.environ['BAPO_LOCAL_RETRIEVER_PID'])
if identity['pid']!=pid:raise RuntimeError('foreign loopback retriever')
env=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
for k in ('BAPO_RUN_ID','SLURM_JOB_ID'):
 if (k+'='+os.environ[k]).encode() not in env:raise RuntimeError('foreign retriever run/job')
PY
# Short, exclusive, job-owned socket directory; retain its recorded path on exit.
RUNTIME=$("$ROOT/env/bin/python" -B - <<'PY'
import tempfile,os,json
from pathlib import Path
p=tempfile.mkdtemp(prefix='bapo-aligned-'+os.environ['SLURM_JOB_ID']+'-');os.chmod(p,0o700)
with (Path(os.environ['BAPO_RUN_ID'])/'ray-runtime-path.json').open('x') as f:json.dump(dict(path=p,run=os.environ['BAPO_RUN_ID'],job=os.environ['SLURM_JOB_ID'],scope='OWN_SHORT_RAY_SOCKET_DIRECTORY'),f)
print(p)
PY
)
export SEARCH_R1_BASELINE_RUN_ROOT="$RUN" RAY_ADDRESS=127.0.0.1:16383
# All production bindings are exported before head creation; no global ray stop.
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/ray_cli_compat.py" start --head --block --node-ip-address=127.0.0.1 --port=16383 --min-worker-port=20000 --max-worker-port=29999 --num-gpus=8 --num-cpus=24 --object-store-memory=8589934592 --include-dashboard=false --temp-dir="$RUNTIME/ray" > "$RUN/ray-head.log" 2>&1 &
ray_pid=$!
export BAPO_RAY_CLI_PID="$ray_pid"
ready=false
for ((i=0;i<60;i++)); do
 kill -0 "$ray_pid"
 if "$ROOT/env/bin/python" -B -c 'import socket;socket.create_connection(("127.0.0.1",16383),timeout=1).close()' 2>/dev/null; then ready=true;break;fi
 sleep 2
done
[[ "$ready" = true ]]
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_service_proof.py" ray-freeze
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_driver.py" > "$RUN/train.log" 2> "$RUN/driver-time.log"

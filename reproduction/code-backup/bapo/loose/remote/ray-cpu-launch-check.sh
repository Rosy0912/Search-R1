#!/bin/bash
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
RUN=$R/runs/ray-cli-cpu-check-20261003-retry1
mkdir "$RUN"
ln -s "$RUN" /tmp/bapo-ray-cpu-check-20261003-retry1
export BAPO_RUN_ID="$RUN" CUDA_VISIBLE_DEVICES='' RAY_USAGE_STATS_ENABLED=0 OMP_NUM_THREADS=1
unset RAY_ADDRESS http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
trap '$R/env/bin/python $R/tools/owned_processes.py stop --run "$RUN" > "$RUN/cleanup.json"' EXIT
$R/env/bin/python $R/tools/ray_cli_compat.py start --head --block --port=16383 --min-worker-port=20000 --max-worker-port=29999 --num-gpus=0 --num-cpus=1 --object-store-memory=134217728 --include-dashboard=false --temp-dir=/tmp/bapo-ray-cpu-check-20261003-retry1/ray > "$RUN/head.log" 2>&1 &
head=$!
for ((i=0;i<30;i++)); do
 kill -0 "$head"
 if $R/env/bin/python -c 'import socket; socket.create_connection(("127.0.0.1",16383),timeout=1).close()' 2>/dev/null; then break;fi
 sleep 2
done
$R/env/bin/python - <<'PY'
import ray
ray.init(address='127.0.0.1:16383')
resources=ray.cluster_resources()
assert resources.get('GPU',0)==0,resources
@ray.remote
def cpu_probe():return 6*7
assert ray.get(cpu_probe.remote())==42
print('RAY_CPU_HEAD_TASK_PASS',resources,flush=True)
ray.shutdown()
PY

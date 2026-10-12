#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
test "${SLURM_JOB_NODELIST:?}" = h100-9
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/repo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost,10.32.0.11 no_proxy=127.0.0.1,localhost,10.32.0.11
for attempt in $(seq 1 120); do
 if "$ROOT/env/bin/python" -c 'import urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({})).open("http://10.32.0.11:18082/provenance",timeout=5).read()' >/dev/null 2>&1; then break; fi
 sleep 5
done
exec "$ROOT/env/bin/python" "$ROOT/tools/probe_h100_gpu_retrieval_http.py" --baseline "$ROOT/runs/retrieval-2854/cpu.json" --output "$ROOT/logs/h100-native-http-$SLURM_JOB_ID.json"

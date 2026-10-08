#!/bin/bash
set -euo pipefail
root=/dataset/shaoshitong/search-r1-rl/bapo
export PATH="$root/env/bin:$root/tools:$PATH"
while [ ! -f "$root/logs/environment.ready" ]; do sleep 20; done
bash "$root/tools/cpu-gate.sh"
export UV_CACHE_DIR="$root/cache/uv" SEARCH_R1_BASE="$root"
name=faiss_gpu_cu12-1.8.0.2-cp310-cp310-manylinux_2_28_x86_64.whl
while [ ! -f "$root/wheelhouse/$name" ]; do sleep 20; done
printf '%s  %s\n' 7ea5e6aa82897fd0c1af3b1decc9c8659a4398fed42de1d7746715b64bbf9bf9 "$root/wheelhouse/$name" | sha256sum -c -
exec 8>"$root/gpu-job.lock"; flock -n 8
uv pip install --python "$root/env/bin/python" --target "$root/retriever-gpu-lib" --no-deps "$root/wheelhouse/$name"
flock -u 8
while ! /usr/bin/python3.10 -c 'import json,sys;sys.exit(not json.load(open("/dataset/shaoshitong/search-r1-rl/bapo/assets-root/logs/assets-status.json"))["complete"])' 2>/dev/null; do sleep 20; done
bash "$root/tools/prepare.sh"

#!/bin/bash
set -euo pipefail
root=/dataset/shaoshitong/search-r1-rl/bapo
export PATH="$root/env/bin:$root/tools:$PATH" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SEARCH_R1_ASSETS="$root/assets-root"
cd "$root/repo"
exec 8>"$root/gpu-job.lock"; flock -n 8
http_proxy=http://10.33.0.16:3128 https_proxy=http://10.33.0.16:3128 UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=2 UV_CACHE_DIR="$root/cache/uv" uv pip install --python "$root/env/bin/python" --index-url https://pypi.org/simple/ 'pytest==8.4.2'
python reproduction/h20/check_environment.py > "$root/logs/environment-acceptance.json"
uv pip freeze --python "$root/env/bin/python" > "$root/logs/environment-freeze.txt"
flock -u 8
PYTHONPATH=.:tests/baselines python "$root/tools/cpu-tests.py" > "$root/logs/cpu-full.txt" 2>&1
python - <<'PYCODE'
import json,os,sys
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo')
sys.path.insert(0,str(root/'repo/reproduction/h20'))
from download_assets import valid,BASE
assets=[a for a in json.loads((BASE/'assets.json').read_text()) if a['path'].startswith('qwen2.5-3b/')]
assert len(assets)==9
assert all(valid(BASE/'assets'/a['path'],a) for a in assets)
(root/'logs/base-model-verification.json').write_text(json.dumps({'verified':True,'files':assets},indent=2))
PYCODE
git rev-parse HEAD > "$root/logs/cpu-gate-source.txt"
touch "$root/logs/cpu-gate.ready"

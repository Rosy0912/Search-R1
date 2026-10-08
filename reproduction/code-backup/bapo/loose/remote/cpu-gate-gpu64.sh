#!/bin/bash
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
SOURCE=b20f371972bb1bc41b735c6cd350b9a779d8820d
export PATH="$R/env/bin:$R/tools:$PATH" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$R/repo"
exec 8> "$R/gpu-job.lock"
flock -n 8
test "$(git rev-parse HEAD)" = "$SOURCE"
test -z "$(git status --porcelain)"
python reproduction/h20/check_environment.py > "$R/logs/environment-check-b20f371.json"
python - <<'PY'
import hashlib
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo/logs')
assert hashlib.sha256((r/'environment-check-b20f371.json').read_bytes()).digest()==hashlib.sha256((r/'environment-acceptance.json').read_bytes()).digest(), 'Environment changed'
PY
PYTHONPATH=.:tests/baselines python "$R/tools/cpu-tests.py" > "$R/logs/cpu-full-b20f371.txt" 2>&1
cp "$R/logs/cpu-gate-source.txt" "$R/logs/cpu-gate-source-before-b20f371.txt"
git rev-parse HEAD > "$R/logs/cpu-gate-source.txt"
touch "$R/logs/cpu-gate.ready" "$R/logs/cpu-gate-b20f371.ready"
tail -n 3 "$R/logs/cpu-full-b20f371.txt"

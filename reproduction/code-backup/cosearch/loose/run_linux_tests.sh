#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/cosearch
export TMPDIR="$AGENT_ROOT/tmp" UV_CACHE_DIR="$AGENT_ROOT/cache/uv"
export HF_HOME="$AGENT_ROOT/cache/hf" XDG_CACHE_HOME="$AGENT_ROOT/cache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
trap 'printf "%s\n" "$?" > "$AGENT_ROOT/logs/linux-cpu-exit-code.txt"' EXIT
/root/.local/bin/uv pip install --python "$AGENT_ROOT/env/bin/python" \
    --target "$AGENT_ROOT/tools/test-lib" --index-url https://mirrors.aliyun.com/pypi/simple/ \
    -c "$AGENT_ROOT/repo/reproduction/h20/constraints-h20.txt" 'pytest==8.4.2'
cd "$AGENT_ROOT/repo"
export PYTHONPATH="$AGENT_ROOT/tools/test-lib:$AGENT_ROOT/repo"
"$AGENT_ROOT/env/bin/python" -m pytest tests/baselines tests/reproduction -q

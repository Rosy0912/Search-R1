#!/bin/bash
# Run on the Linux H20 target after copying this project's scripts beside this file.
set -euo pipefail
ulimit -c 0
BASE=${SEARCH_R1_BASE:-/data/search-r1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${SEARCH_R1_PYTHON:-python3.10}
UPSTREAM_COMMIT=598e61bd1d36895726d28a8d06b3a15bed19f5d3
FLASH_NAME=flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
FLASH_SHA=a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440
FLASH_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FLASH_NAME
for executable in uv "$PYTHON" flock curl sha256sum git; do command -v "$executable" >/dev/null; done
"$PYTHON" -c 'import platform,sys; assert platform.system()=="Linux" and platform.machine()=="x86_64" and sys.version_info[:2]==(3,10), "Requires Linux x86_64 Python 3.10"'
mkdir -p "$BASE"/{logs,cache/uv,cache/pip,tmp,assets,wheelhouse}
# Same outer lock as the pipeline: never mutate an environment while its job runs.
exec 9>"$BASE/gpu-job.lock"
flock -w 10 9 || { echo 'Project job lock occupied; stop your own job before environment installation.' >&2; exit 1; }
export TMPDIR="$BASE/tmp" UV_CACHE_DIR="$BASE/cache/uv" PIP_CACHE_DIR="$BASE/cache/pip"
export HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache" VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
test -d "$BASE/repo/.git" || { echo "Copy the reviewed repository to $BASE/repo first (see environment-setup.md); no automatic branch checkout." >&2; exit 1; }
git -C "$BASE/repo" cat-file -e "$UPSTREAM_COMMIT^{commit}"
git -C "$BASE/repo" rev-parse HEAD > "$BASE/logs/environment-source-commit.txt"
git -C "$BASE/repo" diff --stat > "$BASE/logs/environment-source-diff.txt"
if [ ! -x "$BASE/env/bin/python" ]; then
  uv venv --python "$PYTHON" "$BASE/env"
fi
"$BASE/env/bin/python" -c 'import sys; assert sys.version_info[:2]==(3,10), "Existing env has wrong Python; use a new project directory"'
# Invalidate only our old marker; failures must not leave a stale ready signal.
if [ -f "$BASE/logs/environment.ready" ]; then
  mv "$BASE/logs/environment.ready" "$BASE/logs/environment.ready.previous"
fi
INSTALL_ARGS=(--index-url "${SEARCH_R1_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}" --find-links "$BASE/wheelhouse")
if [ "${BOOTSTRAP_OFFLINE:-0}" = 1 ]; then
  INSTALL_ARGS=(--no-index --find-links "$BASE/wheelhouse" --offline)
fi
# First resolve Torch's normal CUDA12.1 dependencies. Do not put cuBLAS12.4 in
# this resolver transaction: it conflicts with Torch2.4's declared exact pin.
uv pip install --python "$BASE/env/bin/python" "${INSTALL_ARGS[@]}" \
  -r "$SCRIPT_DIR/requirements-h20.txt" -c "$SCRIPT_DIR/constraints-h20.txt"
FLASH_PATH="$BASE/assets/$FLASH_NAME"
check_flash(){ printf '%s  %s\n' "$FLASH_SHA" "$1" | sha256sum -c - >/dev/null; }
if [ -f "$FLASH_PATH" ] && ! check_flash "$FLASH_PATH"; then
  mv "$FLASH_PATH" "$FLASH_PATH.invalid.$(date +%s)"
fi
if [ ! -f "$FLASH_PATH" ]; then
  if [ "${BOOTSTRAP_OFFLINE:-0}" = 1 ]; then
    echo "Missing verified FlashAttention wheel. Follow environment-setup.md local relay instructions; failed copies are preserved." >&2
    exit 1
  fi
  # Never discard a previous partial. Start a new attempt in an independent file.
  PARTIAL="$FLASH_PATH.partial.$(date +%s).$$"
  if ! curl -fLsS --connect-timeout 15 --max-time 1800 --speed-time 60 --speed-limit 1024 \
       -o "$PARTIAL" "$FLASH_URL" || ! check_flash "$PARTIAL"; then
    echo "Direct wheel download failed; partial retained at $PARTIAL. Use Mac relay in environment-setup.md, then rerun." >&2
    exit 1
  fi
  mv "$PARTIAL" "$FLASH_PATH"
fi
check_flash "$FLASH_PATH"
uv pip install --python "$BASE/env/bin/python" --no-deps "$FLASH_PATH"
uv pip install --python "$BASE/env/bin/python" --no-build-isolation --no-deps -e "$BASE/repo"
# Last install: deliberate H20 cuBLAS workaround, no resolver may alter Torch.
uv pip install --python "$BASE/env/bin/python" "${INSTALL_ARGS[@]}" --no-deps 'nvidia-cublas-cu12==12.4.5.8'
"$BASE/env/bin/python" "$SCRIPT_DIR/check_environment.py" \
  > "$BASE/logs/environment-acceptance.json"
uv pip freeze --python "$BASE/env/bin/python" > "$BASE/logs/environment-freeze.txt"
uv pip check --python "$BASE/env/bin/python" > "$BASE/logs/environment-dependency-check.txt" 2>&1 || true
# check_environment permits exactly the recorded Torch/cuBLAS exception and
# rejects other missing/mismatched dependencies. This marker means imports and
# metadata pass; the pipeline still must run NCCL/BF16/inference/short training.
touch "$BASE/logs/environment.ready"
echo 'ENVIRONMENT_IMPORTS_AND_DECLARED_EXCEPTION_OK; GPU integration remains mandatory.'

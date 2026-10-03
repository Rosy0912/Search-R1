#!/bin/bash
set -euo pipefail
BASE=${1:?task root}
SOURCE_PYTHON=/home/shaoshitong/miniconda3/envs/EditVerse/bin/python
mkdir -p "$BASE/logs" "$BASE/wheels" "$BASE/receipts" "$BASE/tmp" "$BASE/sources"
export TMPDIR="$BASE/tmp"
if [[ ! -x "$BASE/env/bin/python" ]]; then
 "$SOURCE_PYTHON" -m venv "$BASE/env"
fi
export PIP_CACHE_DIR="$BASE/pip-cache"
PYTHON="$BASE/env/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel setuptools==79.0.1
# The current PyPI namesake lacks pyairports.airports; use original pinned project.
AIRPORT_COMMIT=9699eae8c6b03c97239018dea92e01daa8594936
curl -fL --connect-timeout 15 --retry 2 --max-time 180 "https://codeload.github.com/NICTA/pyairports/tar.gz/$AIRPORT_COMMIT" -o "$BASE/sources/pyairports-$AIRPORT_COMMIT.tar.gz"
sha256sum "$BASE/sources/pyairports-$AIRPORT_COMMIT.tar.gz" > "$BASE/receipts/pyairports-source.sha256"
"$PYTHON" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$BASE/wheels" "$BASE/sources/pyairports-$AIRPORT_COMMIT.tar.gz"
"$PYTHON" -m pip download --only-binary=:all: --no-binary=antlr4-python3-runtime --find-links "$BASE/wheels" -r "$BASE/source/experiments/search_e1/control/requirements-hkust.txt" -d "$BASE/wheels"
for source in "$BASE"/wheels/*.tar.gz; do
 "$PYTHON" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$BASE/wheels" "$source"
done
"$PYTHON" -m pip install --no-index --find-links "$BASE/wheels" -r "$BASE/source/experiments/search_e1/control/requirements-hkust.txt"
FLASH=flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
curl -fL --connect-timeout 15 --retry 2 --max-time 600 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/${FLASH/+/%2B}" -o "$BASE/wheels/$FLASH.part"
mv "$BASE/wheels/$FLASH.part" "$BASE/wheels/$FLASH"
"$PYTHON" -m pip install --no-deps "$BASE/wheels/$FLASH"
"$PYTHON" -m pip check
"$PYTHON" -c "import torch, transformers, vllm, flash_attn, tensordict, ray, faiss; from pyairports.airports import AIRPORT_LIST; print('IMPORT_VALIDATED', torch.__version__, vllm.__version__, len(AIRPORT_LIST))"
"$PYTHON" -m pip freeze > "$BASE/receipts/environment-freeze.txt"
sha256sum "$BASE"/wheels/* > "$BASE/receipts/wheels.sha256"
touch "$BASE/receipts/env.ready"

#!/bin/bash
set -euo pipefail
BASE=${1:?task root}
SOURCE_PYTHON=/home/shaoshitong/miniconda3/envs/EditVerse/bin/python
mkdir -p "$BASE/logs" "$BASE/wheels" "$BASE/receipts" "$BASE/tmp"
export TMPDIR="$BASE/tmp"
if [[ ! -x "$BASE/env/bin/python" ]]; then
 "$SOURCE_PYTHON" -m venv "$BASE/env"
fi
export PIP_CACHE_DIR="$BASE/pip-cache"
PYTHON="$BASE/env/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel setuptools
"$PYTHON" -m pip download --only-binary=:all: --no-binary=antlr4-python3-runtime,pyairports -r "$BASE/source/experiments/search_e1/control/requirements-hkust.txt" -d "$BASE/wheels"
for source in "$BASE"/wheels/*.tar.gz; do
 "$PYTHON" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$BASE/wheels" "$source"
done
"$PYTHON" -m pip install --no-index --find-links "$BASE/wheels" -r "$BASE/source/experiments/search_e1/control/requirements-hkust.txt"
FLASH=flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
curl -fL --connect-timeout 15 --retry 2 --max-time 600 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/${FLASH/+/%2B}" -o "$BASE/wheels/$FLASH.part"
mv "$BASE/wheels/$FLASH.part" "$BASE/wheels/$FLASH"
"$PYTHON" -m pip install --no-deps "$BASE/wheels/$FLASH"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$BASE/receipts/environment-freeze.txt"
sha256sum "$BASE"/wheels/* > "$BASE/receipts/wheels.sha256"
touch "$BASE/receipts/env.ready"

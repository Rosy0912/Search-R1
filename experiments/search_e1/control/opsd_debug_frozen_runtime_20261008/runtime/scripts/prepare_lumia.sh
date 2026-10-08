#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/upstream/Search-R1:$ROOT"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  /home/jmzhang/miniconda3/envs/searchr1/bin/python -m venv --system-site-packages "$ROOT/.venv"
fi
PYTHON="$ROOT/.venv/bin/python"
"$ROOT/.venv/bin/pip" install --no-deps peft==0.13.2 rapidfuzz==3.13.0
if ! "$PYTHON" -c 'import importlib.metadata as m; assert m.version("flash_attn")'; then
  "$ROOT/.venv/bin/pip" install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl'
fi
bash scripts/bootstrap.sh
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='models/Qwen2.5-3B-Instruct')
PY
"$PYTHON" -m search_e1.prepare --output data

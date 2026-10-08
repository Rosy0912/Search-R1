#!/bin/bash
# Independent wheel overlay: does not replace the trainer's faiss-cpu or NumPy.
set -euo pipefail
BASE=${SEARCH_R1_BASE:-/data/search-r1}
NAME=faiss_gpu_cu12-1.8.0.2-cp310-cp310-manylinux_2_28_x86_64.whl
SHA=7ea5e6aa82897fd0c1af3b1decc9c8659a4398fed42de1d7746715b64bbf9bf9
mkdir -p "$BASE/wheelhouse" "$BASE/retriever-gpu-lib"
exec 9>"$BASE/gpu-job.lock"
flock -n 9 || { echo 'Project job is active; do not mutate its retriever overlay.' >&2; exit 2; }
WHEEL="$BASE/wheelhouse/$NAME"
if [ ! -f "$WHEEL" ]; then
  URL=$(python3 -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("https://pypi.org/pypi/faiss-gpu-cu12/1.8.0.2/json",timeout=15)); print(next(f["url"] for f in d["urls"] if f["filename"]=="faiss_gpu_cu12-1.8.0.2-cp310-cp310-manylinux_2_28_x86_64.whl"))')
  curl -fL --connect-timeout 15 --max-time 120 -o "$WHEEL.partial" "$URL" || {
    echo "Remote wheel download stalled: relay this exact wheel from the Mac and verify SHA256 $SHA; retain any failed relay until verified." >&2
    exit 2
  }
  printf '%s  %s\n' "$SHA" "$WHEEL.partial" | sha256sum -c -
  mv "$WHEEL.partial" "$WHEEL"
fi
printf '%s  %s\n' "$SHA" "$WHEEL" | sha256sum -c -
UV_CACHE_DIR="$BASE/cache/uv" uv pip install --python "$BASE/env/bin/python" --target "$BASE/retriever-gpu-lib" --no-deps "$WHEEL"
PYTHONPATH="$BASE/retriever-gpu-lib" "$BASE/env/bin/python" - <<'PY'
import torch,faiss,numpy
assert numpy.__version__=='1.26.4'
assert faiss.__version__=='1.8.0' and faiss.get_num_gpus()==8
print('ISOLATED_GPU_FAISS_IMPORT_OK',faiss.__file__)
PY

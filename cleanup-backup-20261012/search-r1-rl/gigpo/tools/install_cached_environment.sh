#!/bin/bash
set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
test -f "$ROOT/logs/cluster-copy-20261003.exit"
test "$(cat "$ROOT/logs/cluster-copy-20261003.exit")" = 0
exec 9>"$ROOT/gpu-job.lock"
flock -n 9
export PATH="$ROOT/tools:$PATH" UV_CACHE_DIR="$ROOT/cache/uv" TMPDIR="$ROOT/tmp"
if [ -d "$ROOT/env" ]; then mv "$ROOT/env" "$ROOT/env.before-cluster-$(date +%s)"; fi
uv venv --python /usr/bin/python3.10 "$ROOT/env"
cp -a "$ROOT/cache/source-site-packages/." "$ROOT/env/lib/python3.10/site-packages/"
find "$ROOT/env/lib/python3.10/site-packages" -maxdepth 1 \( -name '__editable__*verl*' -o -name 'verl.egg-link' -o -name 'verl*.dist-info' \) -exec rm -rf {} +
uv pip install --python "$ROOT/env/bin/python" --no-build-isolation --no-deps --offline -e "$ROOT/repo"
uv pip install --python "$ROOT/env/bin/python" --no-deps "$ROOT/wheelhouse/click-8.2.1-py3-none-any.whl"
export CUDA_VISIBLE_DEVICES="" SEARCH_R1_BASE="$ROOT" SEARCH_R1_REPO="$ROOT/repo" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$ROOT/env/bin/python" "$ROOT/repo/reproduction/h20/check_environment.py" > "$ROOT/logs/environment-acceptance.json"
uv pip freeze --python "$ROOT/env/bin/python" > "$ROOT/logs/environment-freeze.txt"
uv pip check --python "$ROOT/env/bin/python" > "$ROOT/logs/environment-dependency-check.txt" 2>&1 || true
cat > "$ROOT/env/bin/ray" <<'RAY'
#!/bin/bash
exec "$(dirname "$0")/python" -m ray.scripts.scripts "$@"
RAY
chmod +x "$ROOT/env/bin/ray"
"$ROOT/env/bin/ray" --version
touch "$ROOT/logs/environment.ready"
echo CACHED_ENVIRONMENT_STRICT_GATE_PASS

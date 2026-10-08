#!/bin/bash
set -euo pipefail
root=/dataset/shaoshitong/search-r1-rl/topd
trap 'rc=$?; printf "%s\n" "$rc" > "$root/logs/cache-environment.exit"' EXIT
while [ ! -f "$root/logs/cluster-cache-copy.exit" ]; do sleep 15; done
test "$(cat "$root/logs/cluster-cache-copy.exit")" = 0
exec 9>"$root/gpu-job.lock"
flock -w 60 9
export PATH="$root/tools:$PATH" UV_CACHE_DIR="$root/cache/uv" TMPDIR="$root/tmp"
if [ -f "$root/logs/environment.ready" ]; then mv "$root/logs/environment.ready" "$root/logs/environment.ready.before-cache-copy"; fi
uv venv --python /usr/bin/python3.10 --clear "$root/env"
cp -a --reflink=auto "$root/cache/source-site-packages/." "$root/env/lib/python3.10/site-packages/"
find "$root/env/lib/python3.10/site-packages" -maxdepth 1 \( -name '__editable__*verl*' -o -name 'verl.egg-link' -o -name 'verl*.dist-info' \) -exec rm -rf {} +
uv pip install --python "$root/env/bin/python" --no-build-isolation --no-deps --offline -e "$root/repo"
export CUDA_VISIBLE_DEVICES="" SEARCH_R1_BASE="$root" SEARCH_R1_REPO="$root/repo" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$root/env/bin/python" "$root/repo/reproduction/h20/check_environment.py" > "$root/logs/environment-acceptance.json"
uv pip freeze --python "$root/env/bin/python" > "$root/logs/environment-freeze.txt"
uv pip check --python "$root/env/bin/python" > "$root/logs/environment-dependency-check.txt" 2>&1 || true
git -C "$root/repo" rev-parse HEAD > "$root/logs/environment-source-commit.txt"
touch "$root/logs/environment.ready"
printf 'ENVIRONMENT_PINNED_IMPORTS_PASS\n'

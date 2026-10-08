set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
A=$R/author-native
mkdir -p "$A/logs" "$A/tools" "$A/cache"
export PIP_CACHE_DIR="$A/cache/pip"
if [ ! -d "$A/repo/.git" ]; then
 timeout 60 git clone https://github.com/snap-research/CoSearch.git "$A/repo"
fi
test "$(git -C "$A/repo" rev-parse HEAD)" = 763bf8c4d7caa363ad845d39ddfd53b81ae377bd
git -C "$A/repo" status --short
"$R/env/bin/python" -m pip install --target "$A/bootstrap" uv --timeout 15 --retries 0
PYTHONPATH="$A/bootstrap" "$R/env/bin/python" -m uv --version

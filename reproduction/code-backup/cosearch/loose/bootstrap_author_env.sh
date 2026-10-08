set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
A=$R/author-native
export UV_INSTALL_DIR="$A/bootstrap/bin"
export UV_NO_MODIFY_PATH=1
curl -fsSL --connect-timeout 15 --max-time 60 https://astral.sh/uv/install.sh -o "$A/tools/uv-install.sh"
sh "$A/tools/uv-install.sh"
export UV_CACHE_DIR="$A/cache/uv"
export UV_PYTHON_INSTALL_DIR="$A/python"
"$A/bootstrap/bin/uv" venv --python 3.11 "$A/env"

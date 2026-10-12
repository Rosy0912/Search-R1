#!/bin/bash
set -euo pipefail
export SEARCH_R1_BASE=/mnt/data/shaoshitong/search-r1-rl/gigpo
export SEARCH_R1_PYTHON=/usr/bin/python3.10
export PATH="$SEARCH_R1_BASE/tools:$PATH"
export HTTP_PROXY=http://127.0.0.1:18971 HTTPS_PROXY=http://127.0.0.1:18971 http_proxy=http://127.0.0.1:18971 https_proxy=http://127.0.0.1:18971
export UV_HTTP_TIMEOUT=60 UV_CONCURRENT_DOWNLOADS=4
bash "$SEARCH_R1_BASE/repo/reproduction/h20/bootstrap.sh"

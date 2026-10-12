#!/bin/bash
set -euo pipefail
export SEARCH_R1_ASSETS=/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root
export HTTP_PROXY=http://127.0.0.1:18971 HTTPS_PROXY=http://127.0.0.1:18971 http_proxy=http://127.0.0.1:18971 https_proxy=http://127.0.0.1:18971
python3 /mnt/data/shaoshitong/search-r1-rl/gigpo/repo/reproduction/h20/download_assets.py

#!/usr/bin/env bash
set -euo pipefail
root=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_BASE=$root SEARCH_R1_ASSETS=$root/assets-root SEARCH_R1_REPO=$root/repo
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
source "$root/env/bin/activate"
while [[ ! -f "$root/logs/download-assets-20261003-proxy.exit" ]]; do sleep 30; done
while [[ ! -f "$root/logs/wiki-http1-20261003.exit" ]]; do sleep 30; done
# Every actual file is verified by prepare_assets; original run may record a repaired HTTP2 failure.
python -u "$root/tools/runtime/prepare_assets.py"

#!/usr/bin/env bash
set -euo pipefail
root=/mnt/data/shaoshitong/search-r1-rl/opsd
export SEARCH_R1_BASE=$root SEARCH_R1_ASSETS=$root/assets-root SEARCH_R1_REPO=$root/repo
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
source "$root/env/bin/activate"
for label in aa gz ab hotpot; do
    marker="$root/logs/parallel-v2-$label-20261003.exit"
    while [[ ! -f "$marker" ]]; do sleep 30; done
    [[ "$(cat "$marker")" == 0 ]]
done
python -u "$root/tools/runtime/prepare_assets.py"

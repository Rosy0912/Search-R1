#!/usr/bin/env bash
set -euo pipefail
export SEARCH_G1_BASE=/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011
export SEARCH_G1_SOURCE_COMMIT=2de313e3e455e005e0d7db83104942bad8bd977a
export SEARCH_G1_ALLOW_EXISTING_GPU_PROCESSES=1
export SEARCH_G1_ROLLOUT_MEMORY_UTILIZATION=0.1
export SEARCH_G1_LOCAL_RETRIEVAL_ASSETS=/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki
exec bash /mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/repo/reproduction/common8/launch_native.sh

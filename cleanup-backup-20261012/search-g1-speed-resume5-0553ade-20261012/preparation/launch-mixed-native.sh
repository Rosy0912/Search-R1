#!/usr/bin/env bash
set -euo pipefail
export SEARCH_G1_FULL_STATE_RESTORE_RECEIPT=/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/diagnostics/checkpoint5-cpu-validation-once-v1/RESULT.json
export SEARCH_G1_EXECUTION_PROFILE=native8_bf16_varlen_fp32_master
export SEARCH_G1_SLURM_SHARED_AUTHORIZATION=/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/preparation/shared-slurm-authorization.json
export SEARCH_G1_BASE=/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012
export SEARCH_G1_SOURCE_COMMIT=0553adec146537e07b84377ce31cba41a54f37cd
export SEARCH_G1_ALLOW_EXISTING_GPU_PROCESSES=1
export SEARCH_G1_ROLLOUT_MEMORY_UTILIZATION=0.1
export SEARCH_G1_LOCAL_RETRIEVAL_ASSETS=/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki
exec bash /mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/repo/reproduction/common8/launch_native.sh

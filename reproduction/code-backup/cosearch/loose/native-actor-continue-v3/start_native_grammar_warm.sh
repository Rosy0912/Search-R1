#!/usr/bin/env bash
set -euo pipefail
BASE=/dataset/shaoshitong/search-r1-rl/cosearch
test "${SLURM_JOB_NODELIST:?}" = h100-16
test "${SLURM_NNODES:?}" = 1
exec env PYTHONPATH="$BASE/repo:$BASE/author-native/native-actor-continue-tools-v3" "$BASE/env/bin/python" -m search_r1.baselines.preflight -- "$BASE/env/bin/python" "$BASE/author-native/native-actor-continue-tools-v3/native_grammar_warm_job.py"

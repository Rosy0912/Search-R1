#!/usr/bin/env bash
set -euo pipefail
BASE=/dataset/shaoshitong/search-r1-rl/cosearch
test "${SLURM_JOB_NODELIST:?}" = h100-16
test "${SLURM_NNODES:?}" = 1
# The existing process-scoped lease gate uses only stdlib before exec.
# It checks both 15% thresholds again after taking all UUID flocks.
exec env PYTHONPATH="$BASE/repo" "$BASE/env/bin/python" \
  -m search_r1.baselines.preflight -- \
  "$BASE/author-native/env/bin/python" "$BASE/author-native/tools/native_job.py"

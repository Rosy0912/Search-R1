#!/usr/bin/env bash
set -euo pipefail
root=/mnt/data/shaoshitong/search-r1-rl/opsd
source "$root/env/bin/activate"
cd "$root/repo"
: "${OPSD_EXPECTED_COMMIT:?Pinned source commit required}"
[[ "$(git rev-parse HEAD)" == "$OPSD_EXPECTED_COMMIT" ]]
exec python -m search_r1.baselines.preflight -- bash "$root/tools/runtime/job.sh"

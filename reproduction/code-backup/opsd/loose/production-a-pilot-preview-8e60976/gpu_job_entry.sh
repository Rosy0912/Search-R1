#!/usr/bin/env bash
set -euo pipefail
echo "STAGED PREVIEW: independent review required" >&2
exit 78
root=/mnt/data/shaoshitong/search-r1-rl/opsd
source "$root/env/bin/activate"
cd "$root/repo"
export OPSD_EXPECTED_COMMIT=8e60976d364ab7d7ae51714415813c1359dafe9a
[[ "$(git rev-parse HEAD)" == "$OPSD_EXPECTED_COMMIT" ]]
python "$root/tools/pilot-runtime-8e60976d364a/source_bundle_check.py"
exec python -m search_r1.baselines.preflight -- bash "$root/tools/pilot-runtime-8e60976d364a/job.sh"

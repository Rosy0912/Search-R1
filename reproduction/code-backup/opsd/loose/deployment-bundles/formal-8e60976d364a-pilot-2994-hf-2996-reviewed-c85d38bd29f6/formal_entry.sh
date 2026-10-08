#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Owned allocation required}"
: "${OPSD_STAGE_MANIFEST_SHA256:?Externally reviewed FINAL manifest SHA required}"
: "${OPSD_ACCEPTANCE_RECEIPT:?Populate only after2994 five outers,2996 fresh HF reload and independent acceptance}"
root=/mnt/data/shaoshitong/search-r1-rl/opsd
source "$root/env/bin/activate"
cd "$root/repo"
export OPSD_EXPECTED_COMMIT=8e60976d364ab7d7ae51714415813c1359dafe9a
[[ $(git rev-parse HEAD) == "$OPSD_EXPECTED_COMMIT" ]]
python "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-c85d38bd29f6/verify_formal_prerequisites.py"
exec python -m search_r1.baselines.preflight -- bash "$root/tools/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-c85d38bd29f6/formal_job.sh"

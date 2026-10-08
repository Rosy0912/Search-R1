#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Owned allocation required}"
: "${OPSD_ACCEPTANCE_RECEIPT:?Populate only after2960 five outers,2962 fresh HF reload and independent acceptance}"
root=/mnt/data/shaoshitong/search-r1-rl/opsd
source "$root/env/bin/activate"
cd "$root/repo"
export OPSD_EXPECTED_COMMIT=e43fbfa73088ed70ed3efef79ba561507c0afc33
[[ $(git rev-parse HEAD) == "$OPSD_EXPECTED_COMMIT" ]]
python "$root/tools/formal/verify_formal_prerequisites.py"
exec python -m search_r1.baselines.preflight -- bash "$root/tools/formal/formal_job.sh"

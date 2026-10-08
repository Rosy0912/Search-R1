#!/bin/bash
set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
R=/dataset/shaoshitong/search-r1-rl/bapo
exec 9> "$D/accept5-submit.lock"
flock -n 9
if [ -f "$D/accept5-job-id.txt" ]; then cat "$D/accept5-job-id.txt";exit 0;fi
# No dependency auto-runs: caller must provide successful smoke evidence first.
case "${SMOKE_JOB_ID:?}" in ''|*[!0-9]*) exit 2;; esac
# /dataset is node-local: review artifacts and the verifier live on h100-5.
env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=15 \
  -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-5 \
  "$R/env/bin/python $R/tools/verify_smoke_receipt.py $SMOKE_JOB_ID"
id=$(sbatch --parsable "$D/accept5.sbatch")
printf '%s\n' "$id" > "$D/accept5-job-id.txt"
scontrol show job "$id"

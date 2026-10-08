#!/bin/bash
set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
R=/dataset/shaoshitong/search-r1-rl/bapo
exec 9> "$D/accept5-gpu64-submit.lock"
flock -n 9
if [ -f "$D/accept5-gpu64-job-id.txt" ]; then cat "$D/accept5-gpu64-job-id.txt";exit 0;fi
# No dependency auto-runs: caller must provide successful smoke evidence first.
case "${SMOKE_JOB_ID:?}" in ''|*[!0-9]*) exit 2;; esac
# /dataset is node-local: review artifacts and the verifier live on h100-5.
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 \
  -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-5 \
  "$R/env/bin/python $R/tools/verify_smoke_receipt-gpu64.py $SMOKE_JOB_ID"
existing=$(squeue -h -n bapo-accept5-gpu64 -o '%i')
if [ -n "$existing" ];then printf '%s\n' "$existing" > "$D/accept5-gpu64-job-id.txt";scontrol show job "$existing";exit 0;fi
id=$(sbatch --parsable "$D/accept5-gpu64.sbatch")
printf '%s\n' "$id" > "$D/accept5-gpu64-job-id.txt"
scontrol show job "$id"

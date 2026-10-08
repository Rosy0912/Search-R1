#!/bin/bash
# Head-only idempotent submission; never bypass independent real-five review.
set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
R=/dataset/shaoshitong/search-r1-rl/bapo
case "${ACCEPT5_JOB_ID:?}" in ''|*[!0-9]*) exit 2;; esac
exec 9> "$D/formal500-gpu64-submit.lock"
flock -n 9
if [ -f "$D/formal500-gpu64-job-id.txt" ]; then
 id=$(cat "$D/formal500-gpu64-job-id.txt")
 case "$id" in ''|*[!0-9]*) exit 2;; esac
 printf '%s\n' "$id";exit 0
fi
# This node's /dataset is not h100-5's /dataset. ssh -n preserves caller stdin.
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 \
 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-5 \
 "$R/env/bin/python $R/tools/verify_five_receipt-gpu64.py $ACCEPT5_JOB_ID"
existing=$(squeue -h -n bapo-formal500-gpu64 -o '%i')
if [ -n "$existing" ]; then
 case "$existing" in *[!0-9]*) echo 'Multiple/invalid scheduler matches; manual reconciliation required' >&2;exit 3;; esac
 scontrol show job "$existing"
 printf '%s\n' "$existing" > "$D/formal500-gpu64-job-id.txt";exit 0
fi
# Persist intent BEFORE sbatch: an interrupted/unknown outcome must never re-submit blindly.
if [ -f "$D/formal500-gpu64-submit-intent.txt" ]; then
 echo 'Prior submission outcome unknown; inspect squeue/sacct and reconcile before retry' >&2;exit 3
fi
printf 'accept5=%s\n' "$ACCEPT5_JOB_ID" > "$D/formal500-gpu64-submit-intent.txt"
if ! id=$(sbatch --parsable "$D/formal500-gpu64.sbatch"); then
 echo 'sbatch outcome failed or unknown; intent retained for reconciliation' >&2;exit 3
fi
case "$id" in ''|*[!0-9]*) echo 'Invalid sbatch response; intent retained' >&2;exit 3;; esac
printf '%s\n' "$id" > "$D/formal500-gpu64-job-id.txt"
scontrol show job "$id"

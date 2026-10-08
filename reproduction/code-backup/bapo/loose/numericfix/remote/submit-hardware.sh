#!/bin/bash
set -euo pipefail
D=/dev/shm/bapo-rewardfix-numeric-scheduler-20261004
R=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric
mkdir -p "$D"
exec 9> "$D/hardware-submit.lock"
flock -n 9
if [ -f "$D/hardware-job-id.txt" ]; then
 id=$(cat "$D/hardware-job-id.txt")
 case "$id" in ''|*[!0-9]*) exit 2;; esac
 printf '%s\n' "$id";exit 0
fi
if [ -f "$D/hardware-submit-intent.txt" ]; then
 echo 'Prior submission outcome unknown: reconcile without resubmitting' >&2;exit 3
fi
# The reviewed fixed launch manifest is checked on the worker, not head-local /dataset.
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 \
 "$R/env/bin/python $R/tools/verify_hardware_launch.py"
existing=$(squeue -h -n bapo-rfix-hw-numeric -o '%i')
if [ -n "$existing" ]; then
 echo 'Unexpected existing hardware job; preserve and reconcile' >&2;exit 3
fi
printf 'source=1cceda2f7e201de3f68c876bb9a3b895563c20ce\nphase=hardware\n' > "$D/hardware-submit-intent.txt"
if ! id=$(sbatch --parsable "$D/parity-gpu64.sbatch"); then
 echo 'Unknown/failed submission: intent retained' >&2;exit 3
fi
case "$id" in ''|*[!0-9]*) echo 'Invalid response: intent retained' >&2;exit 3;; esac
printf '%s\n' "$id" > "$D/hardware-job-id.txt"
scontrol show job "$id"

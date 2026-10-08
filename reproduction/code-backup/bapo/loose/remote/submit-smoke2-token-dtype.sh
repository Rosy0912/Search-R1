#!/bin/bash
set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
exec 9> "$D/smoke2-token-dtype-submit.lock"
flock -n 9
if [ -f "$D/smoke2-token-dtype-job-id.txt" ]; then
  cat "$D/smoke2-token-dtype-job-id.txt"
  exit 0
fi
id=$(sbatch --parsable "$D/smoke2-token-dtype.sbatch")
[[ "$id" =~ ^[0-9]+$ ]]
printf '%s\n' "$id" > "$D/smoke2-token-dtype-job-id.txt"
scontrol show job "$id"

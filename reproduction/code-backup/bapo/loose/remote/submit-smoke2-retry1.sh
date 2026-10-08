#!/bin/bash
set -euo pipefail
D=/dev/shm/bapo-scheduler-20261003
R=/dataset/shaoshitong/search-r1-rl/bapo
exec 9> "$D/smoke2-retry1-submit.lock"
flock -n 9
if [ -f "$D/smoke2-retry1-job-id.txt" ]; then cat "$D/smoke2-retry1-job-id.txt";exit 0;fi
sed 's/--job-name=bapo-smoke2/--job-name=bapo-smoke2-retry1/' "$D/smoke2.sbatch" > "$D/smoke2-retry1.sbatch"
id=$(sbatch --parsable "$D/smoke2-retry1.sbatch")
printf '%s\n' "$id" > "$D/smoke2-retry1-job-id.txt"
scontrol show job "$id"

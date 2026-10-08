#!/bin/bash
set -euo pipefail
base=/dev/shm/bapo-scheduler-20261003
exec 9>"$base/submit.lock"
flock -n 9
if [ -s "$base/basic-job-id.txt" ]; then echo 'Existing own submission recorded; no duplicate';exit 0;fi
while ! env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 'test -f /dataset/shaoshitong/search-r1-rl/bapo/logs/environment.ready && test -f /dataset/shaoshitong/search-r1-rl/bapo/logs/cpu-gate.ready'; do sleep 20;done
test "$(env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 'cat /dataset/shaoshitong/search-r1-rl/bapo/logs/cpu-gate-source.txt')" = a3e000de547e1260eadb1a20cb0a350d58c1133d
# Query the unique own job name if a previous unknown submission outcome exists.
existing=$(squeue -h -n bapo-hw-basic-20261003 -o %i)
if [ -n "$existing" ]; then printf '%s\n' "$existing" > "$base/basic-job-id.txt";exit 0;fi
sbatch --parsable --job-name=bapo-hw-basic-20261003 "$base/basic.sbatch" > "$base/basic-job-id.txt"
cat "$base/basic-job-id.txt"

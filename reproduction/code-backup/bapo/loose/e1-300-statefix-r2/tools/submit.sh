#!/bin/bash
set -euo pipefail
D=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-r2-20261004/scheduler-head
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-r2-20261004
phase=${1:?hardware|formal300}
case "$phase" in hardware|formal300) ;; *) exit 2;; esac
exec 8>"$D/submit-global.lock"
flock -n 8
if [ -f "$D/$phase-job-id.txt" ]; then
 id=$(cat "$D/$phase-job-id.txt");case "$id" in ''|*[!0-9]*) exit 2;; esac
 printf '%s\n' "$id";exit 0
fi
if [ -f "$D/$phase-submit-intent.txt" ]; then
 echo 'Unknown prior submission outcome; reconcile the same intent without resubmitting' >&2;exit 3
fi
python3 - "$D" "$phase" <<'PY'
import sys,json,pathlib,hashlib
p=pathlib.Path(sys.argv[1]);m=json.loads((p/'approved-launch-hashes.json').read_text())
for name in ('submit.sh',sys.argv[2]+'.sbatch'):
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==m[name],name
PY
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "$ROOT/env/bin/python $ROOT/tools/verify_launch.py $phase"
if [ "$phase" = formal300 ]; then
 env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "$ROOT/env/bin/python $ROOT/tools/verify_hardware_receipt.py"
fi
active=$(squeue -h -w h100-5 -o '%j' | sed -n '/^bapo/p')
if [ -n "$active" ]; then echo 'An owned BAPO job is still active; no second instance' >&2;exit 3;fi
printf '%s\n' "$phase" > "$D/$phase-submit-intent.txt"
# A second durable copy on the worker survives head scratch loss.
env -u LD_LIBRARY_PATH ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 h100-5 "$ROOT/env/bin/python $ROOT/tools/submission-worker-intent.py $phase"
if ! id=$(sbatch --parsable "$D/$phase.sbatch");then echo 'Submission outcome unknown; preserve intent' >&2;exit 3;fi
case "$id" in ''|*[!0-9]*) echo 'Invalid scheduler response; preserve intent' >&2;exit 3;; esac
printf '%s\n' "$id" > "$D/$phase-job-id.txt"
scontrol show job "$id"

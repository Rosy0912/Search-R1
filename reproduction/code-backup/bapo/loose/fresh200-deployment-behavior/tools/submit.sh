#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006
D=$ROOT/scheduler-head
phase=${1:?phase required}
case "$phase" in diagnostic-discovery128|diagnostic-discovery256|diagnostic-confirmation128|diagnostic-confirmation256|formal200) ;; *) exit 2;; esac
exec 8>"$D/submit-global.lock"
flock -n 8 || { echo 'Another submit transaction holds lock' >&2; exit 3; }
ssh_worker() {
 env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 h100-5 "$@"
}
mirror() {
 local kind=$1
 local suffix=submit-intent; [[ "$kind" = receipt ]] && suffix=receipt
 ssh_worker "env -u LD_LIBRARY_PATH '$ROOT/env/bin/python' -B '$ROOT/tools/launch_transaction.py' 'admit-$kind'" < "$D/$phase-$suffix.json"
}
# No legacy receipt or permission is consulted. The fixed target must pass its new gate.
ssh_worker "cd '$ROOT/repo' && env -u LD_LIBRARY_PATH PYTHONPATH='$ROOT/repo' '$ROOT/env/bin/python' -B '$ROOT/tools/verify-launch.py' '$phase'"
# All BAPO jobs on this node are advisory blockers; durable intents are authoritative.
test -z "$(squeue -h -w h100-5 -o '%j' | sed -n '/[Bb][Aa][Pp][Oo]/p')" || { echo 'BAPO allocation already active' >&2; exit 3; }
result=$(python3 "$ROOT/tools/launch_transaction.py" prepare "$phase")
case "$result" in 'EXISTING '*) mirror intent; mirror receipt; echo "$result"; exit 0;; esac
# Explicit durable worker admission precedes sbatch. SSH failure retains head intent.
mirror intent
# Intent is already fsynced. Any sbatch/response/record failure leaves it unresolved.
id=$(sbatch --no-requeue --parsable --comment="$result" "$ROOT/tools/$phase.sbatch")
[[ "$id" =~ ^[1-9][0-9]*$ ]] || { echo 'Unknown submit outcome; do not retry' >&2; exit 3; }
python3 "$ROOT/tools/launch_transaction.py" record "$phase" "$id"
# Missing acknowledgement leaves worker unresolved and blocks all future phases.
mirror receipt
echo "$id"

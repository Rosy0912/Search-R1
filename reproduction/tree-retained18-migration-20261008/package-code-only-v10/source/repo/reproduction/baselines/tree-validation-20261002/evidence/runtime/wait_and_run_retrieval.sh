#!/bin/bash
set -euo pipefail
CONTROL=/dev/shm/shaoshitong-tree-grpo-control-20261003
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
mkdir -p "$CONTROL"
exec 9>"$CONTROL/retrieval-dispatch.lock"
flock -n 9 || { echo EXISTING_RETRIEVAL_DISPATCHER; exit 1; }
test ! -e "$CONTROL/retrieval-dispatched.marker"
node_check() {
 env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o ControlMaster=auto -o ControlPersist=600 -o "ControlPath=$CONTROL/%h-%p" h100-9 "$1"
}
echo WAITING_FOR_VERIFIED_DERIVED_ASSETS
deadline=$(( $(date +%s) + 86400 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
 if node_check "test -f $ROOT/logs/derived-assets.ready"; then
  date -u +%FT%TZ > "$CONTROL/retrieval-dispatched.marker"
  set +e
  srun -p gpu -w h100-9 -N 1 --gres=gpu:h100:8 --exclusive --ntasks=1 --cpus-per-task=32 --mem=256G --time=02:00:00 --job-name=tree-retrieval-gate bash "$ROOT/tools/real_retrieval_gate.sh"
  code=$?
  echo "$code" > "$CONTROL/retrieval-job.exit"
  exit "$code"
 fi
 if node_check 'test -f /dataset/shaoshitong/search-r1-rl/tree-grpo/logs/direct-copy-supervised-20261003.exit && test "$(cat /dataset/shaoshitong/search-r1-rl/tree-grpo/logs/direct-copy-supervised-20261003.exit)" != 0'; then
  echo ASSET_RETRY_EXHAUSTED_REVIEW_REQUIRED
  echo 1 > "$CONTROL/retrieval-job.exit"
  exit 1
 fi
 sleep 30
done
echo WAIT_TIMEOUT_REVIEW_REQUIRED
echo 1 > "$CONTROL/retrieval-job.exit"
exit 1

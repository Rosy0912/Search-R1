#!/bin/bash
set -euo pipefail
CONTROL=/dev/shm/shaoshitong-tree-grpo-control-20261003
exec 9>"$CONTROL/hybrid-cpu-gate-async-dispatch.lock"
flock -n 9 || exit 1
test ! -e "$CONTROL/hybrid-cpu-gate-async-dispatched.marker"
date -u +%FT%TZ > "$CONTROL/hybrid-cpu-gate-async-dispatched.marker"
set +e
srun -p gpu -w h100-9 -N1 --gres=none --ntasks=1 --cpus-per-task=8 --mem=100G --time=01:00:00 --job-name=tree-hybrid-async-gate-async bash /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/hybrid_cpu_gate_async.sh
rc=$?
echo "$rc" > "$CONTROL/hybrid-cpu-gate-async.exit"
exit "$rc"

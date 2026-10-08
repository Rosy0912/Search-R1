#!/bin/bash
set -euo pipefail
CONTROL=/dev/shm/shaoshitong-tree-grpo-control-20261003
mkdir -p "$CONTROL"
exec 9>"$CONTROL/two-clickfixed-dispatch.lock"
flock -n 9 || exit 1
test ! -e "$CONTROL/two-clickfixed-dispatched.marker"
date -u +%FT%TZ > "$CONTROL/two-clickfixed-dispatched.marker"
set +e
srun -p gpu -w h100-9 -N1 --gres=gpu:h100:8 --exclusive --ntasks=1 --cpus-per-task=32 --mem=256G --time=04:00:00 --job-name=tree-two-updates bash /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/train_tree_stage.sh two
code=$?
echo "$code" > "$CONTROL/two-clickfixed-job.exit"
exit "$code"

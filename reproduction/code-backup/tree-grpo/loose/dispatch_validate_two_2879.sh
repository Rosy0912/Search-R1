#!/bin/bash
set -euo pipefail
CONTROL=/dev/shm/shaoshitong-tree-grpo-control-20261003
exec 9>"$CONTROL/validate-two-2879.lock"
flock -n 9 || exit 1
test ! -e "$CONTROL/validate-two-2879-dispatched.marker"
test "$(cat "$CONTROL/two-hybrid-async-job.exit")" = 0
test -z "$(squeue -h -j 2879 -o %i)"
env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes root@h100-9 'python3 - <<"PY"
import pathlib,json
root=pathlib.Path("/dataset/shaoshitong/search-r1-rl/tree-grpo")
run=root/"runs/tree-two-2879"
assert (root/"logs/two-stage-completed.run").read_text().strip()==str(run)
assert json.loads((run/"cost.json").read_text())["exit_code"]==0
PY'
date -u +%FT%TZ > "$CONTROL/validate-two-2879-dispatched.marker"
set +e
srun -p gpu -w h100-9 -N1 --gres=none --ntasks=1 --cpus-per-task=8 --mem=64G --time=00:45:00 --job-name=tree-two-2879-validate bash /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/validate_cpu_stage.sh two
rc=$?
echo "$rc" > "$CONTROL/validate-two-2879.exit"
exit "$rc"

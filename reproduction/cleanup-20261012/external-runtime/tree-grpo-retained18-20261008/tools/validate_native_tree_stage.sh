#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
STAGE=${1:?}; TRAIN_JOB=${2:?}
case "$STAGE" in two|five|formal) ;; *) exit 2;; esac
[[ "$TRAIN_JOB" =~ ^[1-9][0-9]*$ ]]
test "${SLURM_JOB_NODELIST:?}" = h100-9
test "$(cat "$ROOT/logs/$STAGE-stage-completed.run")" = "$ROOT/runs/tree-$STAGE-$TRAIN_JOB"
exec bash "$ROOT/tools/validate_cpu_stage.sh" "$STAGE"

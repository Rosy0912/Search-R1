#!/usr/bin/env bash
set -euo pipefail
base=/weka_3/shaoshitong/search-e1-config-study-20261004
checkpoint=$base/runs/archive/round-1-grpo/actor/global_step_50
report=$base/receipts/archive-step50-weights.json
test ! -e "$report"
# The original trainer logs step50 after validation and synchronous save.
while ! grep -q 'step:50 -' "$base/runs/archive/round-1-grpo/train.log"; do
  sleep 60
done
test -f "$checkpoint/config.json"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$base/env/bin/python" "$base/source/experiments/search_e1/control/verify_checkpoint.py" \
  --initial "$base/assets/model" --checkpoint "$checkpoint" --report "$report"

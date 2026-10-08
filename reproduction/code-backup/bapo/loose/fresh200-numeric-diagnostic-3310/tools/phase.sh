#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/numeric-diagnostic-3310-20261006
SOURCE_ROOT=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006
python "$ROOT/tools/verify-diagnostic.py" numeric-zero-adam-replay
python -B - <<'PY'
from diagnostic_contract import verify
verify(require_job=True)
PY
export FRESH200_ZERO_ADAM_GUARD_READY=1
exec python -B "$ROOT/tools/numeric-probe.py" --repo "$SOURCE_ROOT/repo" \
 --raw-batch "/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/runs/formal200-6a29642035ff40b2b7787de206f886c0-3307/checkpoints/diagnostics/formal-sampling-attempt_1/formal-first-sampling-batch.pt" \
 --model "$SOURCE_ROOT/assets/qwen2.5-3b-instruct-aa8e725" --output "$BAPO_RUN_ID/numeric-probe" \
 --reference-result "$ROOT/logs/reference3310.actual.json" --prefix-witness "$ROOT/logs/prefix-witness-v2.actual.json" \
 --execute-gpu --approval "$ROOT/logs/diagnostic-approval.json"

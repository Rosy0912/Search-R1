#!/bin/bash
# Read-only acceptance observer. Training never waits for this process.
set -euo pipefail
BASE=/data/search-r1
for ((i=0;i<1440;i++)); do
  if grep -q 'step:5 - .*critic/rewards/mean:' "$BASE/logs/paper-grpo-qwen2.5-3b.log"; then
    if OMP_NUM_THREADS=2 "$BASE/env/bin/python" "$BASE/verify_five_updates.py" \
        > "$BASE/logs/five-updates-acceptance.json.partial" \
        2> "$BASE/logs/five-updates-error.txt"; then
      mv "$BASE/logs/five-updates-acceptance.json.partial" "$BASE/logs/five-updates-acceptance.json"
      echo 'FIVE_REAL_UPDATES_ACCEPTED; trainer continues independently to 500'
      exit 0
    fi
    echo 'Five-step metrics found but acceptance failed; inspect five-updates-error.txt' >&2
    exit 2
  fi
  sleep 30
done
echo 'Five-step observer timed out without an accepted result' >&2
exit 2

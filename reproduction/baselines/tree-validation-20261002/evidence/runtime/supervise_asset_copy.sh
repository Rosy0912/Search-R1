#!/bin/bash
set -u
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
for attempt in $(seq 1 20); do
  printf 'ASSET_RETRY_ATTEMPT %s UTC %s\n' "$attempt" "$(date -u +%FT%TZ)"
  /usr/bin/python3.10 "$ROOT/tools/copy_direct_assets.py"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo 0 > "$ROOT/logs/direct-copy-supervised-20261003.exit"
    exit 0
  fi
  echo "ASSET_ATTEMPT_FAILED $attempt $code; retained own partial files"
  sleep 30
done
echo 1 > "$ROOT/logs/direct-copy-supervised-20261003.exit"
exit 1

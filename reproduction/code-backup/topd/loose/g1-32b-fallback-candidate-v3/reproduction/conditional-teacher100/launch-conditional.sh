#!/usr/bin/env bash
set -euo pipefail
: "${G1_PIPELINE_DIR:?Frozen isolated owned supervisor}"
: "${G1_PIPELINE_CONFIG:?Independent source/native/resource bound config}"
: "${G1_PIPELINE_MODE:?initial-only or train100}"
# Caller is the single outer exclusive8GPU common preflight. Actual FDs are verified.
exec /dataset/shaoshitong/search-r1-rl/topd/env/bin/python "$G1_PIPELINE_DIR/supervise.py" --config "$G1_PIPELINE_CONFIG" --mode "$G1_PIPELINE_MODE"

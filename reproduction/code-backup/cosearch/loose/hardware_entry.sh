#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/cosearch
export PATH="$AGENT_ROOT/env/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TMPDIR="$AGENT_ROOT/tmp"
cd "$AGENT_ROOT/repo"
command -v nvidia-smi
exec python -m search_r1.baselines.preflight -- /bin/bash "$AGENT_ROOT/tools/hardware_body.sh"

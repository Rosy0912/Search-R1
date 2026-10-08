#!/usr/bin/env bash
# Source-fixed newrun entry; allocation/resource supervision remains mandatory.
set -euo pipefail
export BAPO_ALIGNED_CONFIG_NAME=ppo_bapo_baseline_aligned_production
exec bash "$(dirname "$0")/run.sh" "$@"

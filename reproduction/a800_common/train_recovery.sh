#!/usr/bin/env bash
# Distinct trainer entry; the admitted owned recovery pipeline must call this.
# This script does not prepare/claim GPUs or start auxiliary services.
set -euo pipefail
BASE="${SEARCH_G1_BASE:?bound prepared recovery namespace must be explicit}"
export SEARCH_G1_RUN_START=full_state_resume
exec bash "$BASE/repo/reproduction/a800_common/train.sh"

#!/bin/bash
set -euo pipefail
# No implicit package, release, job submission, login-shell environment, or retry.
[[ "$#" -ge 2 && "$1" == "--package" ]] || { echo 'Explicit --package must be first'; exit 2; }
# sbatch copies this script to spool; its dirname is not the immutable package.
PACKAGE="$2"
export PYTHONDONTWRITEBYTECODE=1
exec /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -B "$PACKAGE/start.py" "$@"

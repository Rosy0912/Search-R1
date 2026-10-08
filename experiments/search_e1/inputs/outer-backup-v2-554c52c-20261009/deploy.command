#!/bin/sh
set -eu
# Root runs only after publishing/readback of this input/bootstrap entry.
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B /Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/inputs/outer-backup-v2-554c52c-20261009/deploy_bootstrap.py --apply

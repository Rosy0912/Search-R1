#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/tools:$ROOT/repo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$ROOT/env/bin/python" "$ROOT/tools/probe_validation_cost_configuration.py"

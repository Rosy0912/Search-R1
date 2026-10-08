#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
test "${SLURM_JOB_NODELIST:?}" = h100-9
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/tools:$ROOT/repo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$ROOT/env/bin/python" "$ROOT/tools/probe_validation_cost_ray.py" --output "$ROOT/logs/validation-cost-ray-$SLURM_JOB_ID" --repo "$ROOT/repo"

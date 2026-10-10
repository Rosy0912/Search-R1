#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export E1_RUN_ID=signal-study-eval-recovery-4e1e70d464834408b43cf00e9fd030a5
exec /weka_3/shaoshitong/search-e1-config-study-20261004/env/bin/python /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/evaluation-recovery-31e2d74/opsd_signal_weight_eval_recovery_v2/entry.py --mode study --recovery-ref /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/evaluation-recovery-input-31e2d74.json --recovery-sha256 ff44d73effa3a40ff5ed1f3d6b5049002b6b8937aa491a761b00266d74f1863e -- --study-manifest /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/study-manifest.json --study-manifest-sha256 86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e --resume

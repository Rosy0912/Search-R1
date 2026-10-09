#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export E1_RUN_ID=signal-study-recovery-d96c7fc1e0f14ea2bf3edcb01a22bfa8
exec /weka_3/shaoshitong/search-e1-config-study-20261004/env/bin/python /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/recovery-98b9326/opsd_signal_weight_recovery_v1/entry.py --mode study --recovery-ref /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/recovery-input-98b9326.json --recovery-sha256 73a5594a0dae56125cdb5f7a28f8f108454df5c8c1761d8fa629556b0cac9f4e -- --study-manifest /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/study-manifest.json --study-manifest-sha256 86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e --resume

#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
runtime=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009
reference=/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013
base="${RBA_ABLATION_BASE:?}"
export PATH="$runtime/env/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$source_root:$runtime/retriever-gpu-lib"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RAY_ADDRESS
cd "$source_root"
"$runtime/env/bin/python" -B release/recursive_bran_a/h100_migration/reuse_retriever.py "$reference/retriever"
exec bash "$source_root/release/recursive_bran_a/ablation150_v1/study.sh"

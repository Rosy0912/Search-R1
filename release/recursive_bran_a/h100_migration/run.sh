#!/usr/bin/env bash
set -euo pipefail
base=/dataset/shaoshitong/recursive-bran-a-h100-20261012
source="$(cd "$(dirname "$0")/../../.." && pwd)"
runtime=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009
python="$runtime/env/bin/python"
export RBA_H100_SHARED_ALLOCATION=1
export RECURSIVE_BRAN_BASE="$base" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$runtime/env/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$source:$runtime/retriever-gpu-lib:$source/release/recursive_bran_a/retriever_overlap_candidate:$source/release/recursive_bran_a/h100_migration/oracle_source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TMPDIR="$base/tmp" RAY_TMPDIR="$base/ray" XDG_CACHE_HOME="$base/cache" HF_HOME="$base/cache/hf"
unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME" "$base/receipts" "$base/retriever/receipts"
cd "$source"
"$python" -B release/recursive_bran_a/h100_migration/prepare_train.py > "$base/logs/configure.log" 2>&1
"$python" -B release/recursive_bran_a/h100_migration/prepare_assets.py
export E1_SEGMENTED_PROBE_TOKEN="$(cat "$base/retriever/receipts/private-probe-token")"
"$python" -B -u release/recursive_bran_a/h100_migration/eight_server.py --oracle-source "$source/release/recursive_bran_a/h100_migration/oracle_source" --base "$base/retriever" --assets "$base/retriever/assets" --asset-manifest "$base/retriever/assets/manifest.json" --host 127.0.0.1 --port 18086 > "$base/logs/retriever.log" 2>&1 &
retriever=$!
trap 'kill "$retriever" 2>/dev/null || true; wait "$retriever" 2>/dev/null || true' EXIT
"$python" -B release/recursive_bran_a/h100_migration/activate.py "$base/retriever" "$retriever"
"$python" -B -u release/recursive_bran_a/h100_migration/gate.py --base "$base/retriever" > "$base/logs/retriever-gate.log" 2>&1
"$runtime/env/bin/ray" start --head --node-ip-address=127.0.0.1 --port=16386 --num-gpus=8 --temp-dir="$RAY_TMPDIR" --disable-usage-stats > "$base/logs/ray-start.log" 2>&1
export RAY_ADDRESS=127.0.0.1:16386
"$python" -B -u -m search_r1.baselines.preflight -- "$python" -B -u release/recursive_bran_a/h100_migration/train.py > "$base/logs/training.log" 2>&1

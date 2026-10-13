#!/usr/bin/env bash
set -euo pipefail
base=/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013
source="$base/source-resume100-v1"
runtime=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009
python="$runtime/env/bin/python"
export RBA_H100_ALLOCATION_ID="$SLURM_JOB_ID" RBA_H100_SHARED_ALLOCATION=1
export RECURSIVE_BRAN_BASE="$base" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$runtime/env/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$source:$runtime/retriever-gpu-lib:$source/release/recursive_bran_a/retriever_overlap_candidate:$source/release/recursive_bran_a/h100_migration/oracle_source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0
export CPATH=/root/miniconda3/pkgs/python-3.10.21-h17756b0_0/include/python3.10
export TRITON_LIBCUDA_PATH="$runtime/lib/driver" LIBRARY_PATH="$runtime/lib/driver"
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TMPDIR="$base/tmp" RAY_TMPDIR="/tmp/rba-h100-${RBA_H100_ALLOCATION_ID}" XDG_CACHE_HOME="$base/cache" HF_HOME="$base/cache/hf"
unset LD_LIBRARY_PATH HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME" "$base/logs"
cd "$source"
"$python" -B release/recursive_bran_a/h100_resume100_v1/prepare.py > "$base/logs/configure.log" 2>&1
export RECURSIVE_A_CONTINUATION_BINDING="$base/receipts/continuation-binding.json"
export RECURSIVE_A_CONTINUATION_SHA256="$(sha256sum "$RECURSIVE_A_CONTINUATION_BINDING" | cut -d' ' -f1)"
export E1_SEGMENTED_PROBE_TOKEN="$(cat "$base/retriever/receipts/private-probe-token")"
"$python" -B -u release/recursive_bran_a/h100_migration/eight_server.py --oracle-source "$source/release/recursive_bran_a/h100_migration/oracle_source" --base "$base/retriever" --assets "$base/retriever/assets" --asset-manifest "$base/retriever/assets/manifest.json" --host 127.0.0.1 --port 18086 > "$base/logs/retriever.log" 2>&1 &
retriever=$!
"$python" -B release/recursive_bran_a/h100_migration/activate.py "$base/retriever" "$retriever" > "$base/logs/retriever-activate.log" 2>&1
"$python" -B -u release/recursive_bran_a/h100_migration/gate.py --base "$base/retriever" > "$base/logs/retriever-gate.log" 2>&1
"$python" -B -u -m search_r1.baselines.preflight -- "$python" -B -u release/recursive_bran_a/h100_migration/train.py > "$base/logs/training.log" 2>&1

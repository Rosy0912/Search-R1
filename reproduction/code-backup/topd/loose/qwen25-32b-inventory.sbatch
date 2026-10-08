#!/bin/bash
#SBATCH --job-name=topd32b-official-inventory
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/qwen25-32b-official-inventory-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$AGENT_ROOT/repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/tools/qwen25-32b-eval64"
printf '%s\n' 'b5c80facfb507ed020e29ac48d8246aa025a9967edb12754816717d730c1c27b  build_binding.py' '7380b9d5d675b30abef439e1feaa976325c0a87db539d8b28642ba01da7714dc  eval64.py' | sha256sum -c -
"$AGENT_ROOT/env/bin/python" build_binding.py --official-metadata "$AGENT_ROOT/logs/qwen2.5-32b-official-metadata.json" --data "$AGENT_ROOT/assets-root/data/paper-grpo/validation-64.parquet" --output "$AGENT_ROOT/logs/qwen25-32b-eval64-binding.json"
"$AGENT_ROOT/env/bin/python" eval64.py --binding "$AGENT_ROOT/logs/qwen25-32b-eval64-binding.json" --check-assets-only

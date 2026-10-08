#!/bin/bash
#SBATCH --job-name=topd32b-terminal-fixed413-inventory
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/qwen25-32b-terminal-fixed413-inventory-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$AGENT_ROOT/repo-terminal-fixed413"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/tools/qwen25-32b-terminal-fix-refreeze"
printf '%s\n' '162c2e35a940ad9ece959507a67c71da85e57dafc46cdf434b8c672fff0438d2  build_binding.py' '24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee  eval64.py' | sha256sum -c -
"$AGENT_ROOT/env/bin/python" build_binding.py --official-metadata "$AGENT_ROOT/logs/qwen2.5-32b-official-metadata.json" --data "$AGENT_ROOT/assets-root/data/paper-grpo/validation-64.parquet" --output "$AGENT_ROOT/logs/qwen25-32b-terminal-fixed413-binding.json"
"$AGENT_ROOT/env/bin/python" eval64.py --binding "$AGENT_ROOT/logs/qwen25-32b-terminal-fixed413-binding.json" --check-assets-only

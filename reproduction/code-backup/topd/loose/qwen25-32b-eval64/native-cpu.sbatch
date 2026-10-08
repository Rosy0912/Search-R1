#!/bin/bash
#SBATCH --job-name=qwen25-32b-eval64-contracts
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/qwen25-32b-eval64-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/tools/qwen25-32b-eval64"
sha256sum eval64.py supervisor.py runtime_safety.py test_eval64.py eval64.sbatch native-cpu.sbatch
"$AGENT_ROOT/env/bin/python" -m pytest test_eval64.py -q

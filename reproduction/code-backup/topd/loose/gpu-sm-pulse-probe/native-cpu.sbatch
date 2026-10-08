#!/bin/bash
#SBATCH --job-name=topd-sm-pulse-cpu-contracts
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/gpu-sm-pulse-cpu-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo:$AGENT_ROOT/tools/qwen25-32b-eval64"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/tools"
sha256sum qwen25-32b-eval64/runtime_safety_candidate.py qwen25-32b-eval64/test_runtime_safety_candidate.py gpu-sm-pulse-probe/probe.py gpu-sm-pulse-probe/worker.py gpu-sm-pulse-probe/test_probe.py gpu-sm-pulse-probe/probe.sbatch gpu-sm-pulse-probe/native-cpu.sbatch
"$AGENT_ROOT/env/bin/python" -m pytest qwen25-32b-eval64/test_runtime_safety_candidate.py gpu-sm-pulse-probe/test_probe.py -q

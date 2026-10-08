#!/bin/bash
#SBATCH --job-name=topd-tiny-cuda-sm-telemetry
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/gpu-sm-pulse-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
exec 9>"$AGENT_ROOT/gpu-job.lock";flock -n 9
cd "$AGENT_ROOT/repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTHONPATH="$AGENT_ROOT/repo:$AGENT_ROOT/tools/qwen25-32b-eval64"
sha256sum "$AGENT_ROOT/tools/gpu-sm-pulse-probe/probe.py" "$AGENT_ROOT/tools/gpu-sm-pulse-probe/worker.py" "$AGENT_ROOT/tools/qwen25-32b-eval64/runtime_safety_candidate.py"
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- \
 "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/gpu-sm-pulse-probe/probe.py"

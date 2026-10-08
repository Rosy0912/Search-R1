#!/bin/bash
#SBATCH --job-name=topd-failed-row-fp32-diagnostic
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/parity-diagnostic-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
exec 9>"$AGENT_ROOT/gpu-job.lock";flock -n 9
cd "$AGENT_ROOT/repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTHONPATH="$AGENT_ROOT/repo"
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- \
 "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/topd-parity-diagnostic-wrapper.py"

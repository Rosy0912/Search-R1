#!/bin/bash
#SBATCH --job-name=topd-base-gpu-fp32-score-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/teacher-gpu-fp32-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
cd "$AGENT_ROOT/repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="$AGENT_ROOT/repo"
# Preserve scheduler-selected physicalGPU visibility; never override toGPU0.
# GPU lease covers only this one allocated UUID and survives this exec.
"$AGENT_ROOT/env/bin/python" reproduction/h20/check_environment.py
"$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- \
 "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/topd-teacher-gpu-fp32-probe.py" \
 --output "$AGENT_ROOT/logs/teacher-gpu-fp32-$SLURM_JOB_ID.jsonl"
# Child exited, so CUDA context and UUID lease have released; no global cleanup.
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv

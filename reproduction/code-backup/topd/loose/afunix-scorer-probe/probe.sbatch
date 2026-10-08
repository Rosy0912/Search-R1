#!/bin/bash
#SBATCH --job-name=topd-hardware-afunix8-scorer
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=00:45:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/afunix-scorer-hardware-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
exec 9>"$AGENT_ROOT/gpu-job.lock";flock -n 9
cd "$AGENT_ROOT/repo"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_NO_USAGE_STATS=1 VLLM_ATTENTION_BACKEND=XFORMERS NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTHONPATH="$AGENT_ROOT/repo"
export TMPDIR="$AGENT_ROOT/tmp/afunix-scorer-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$AGENT_ROOT/cache/afunix-scorer-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- \
 "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/afunix-scorer-probe/probe.py"

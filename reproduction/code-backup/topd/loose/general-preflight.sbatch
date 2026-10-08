#!/bin/bash
#SBATCH --job-name=topd-general-preflight
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/general-preflight-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
test -f "$AGENT_ROOT/logs/environment.ready"
test -f "$AGENT_ROOT/logs/full-copy-verification.ready"
exec 9>"$AGENT_ROOT/gpu-job.lock"
flock -n 9
cd "$AGENT_ROOT/repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$AGENT_ROOT/tmp/preflight-$SLURM_JOB_ID" XDG_CACHE_HOME="$AGENT_ROOT/cache/preflight-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
printf 'JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
"$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$AGENT_ROOT/env/bin/python" -m torch.distributed.run --nproc_per_node=8 --master_port=26385 "$AGENT_ROOT/repo/reproduction/h20/cuda_preflight.py"
CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=XFORMERS "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/general-model-preflight.py"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
printf 'GENERAL_HARDWARE_PREFLIGHT_PASS\n'

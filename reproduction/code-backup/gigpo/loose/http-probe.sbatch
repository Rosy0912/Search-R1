#!/bin/bash
#SBATCH --job-name=gigpo-http-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/shaoshitong/search-r1-rl/gigpo/logs/http-probe-%j.log
set -euo pipefail
export AGENT_ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE="$AGENT_ROOT/cache/retriever-datasets"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$AGENT_ROOT/repo"
srun --ntasks=1 "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/http-retrieval-probe.py"

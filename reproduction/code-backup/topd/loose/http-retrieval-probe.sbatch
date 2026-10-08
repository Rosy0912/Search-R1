#!/bin/bash
#SBATCH --job-name=topd-http-retrieval-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:20:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/http-retrieval-probe-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
test -f "$AGENT_ROOT/logs/full-copy-verification.ready"
test -f "$AGENT_ROOT/logs/data-preparation.ready"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$AGENT_ROOT/cache/hf"
cd "$AGENT_ROOT/repo"
"$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/http-retrieval-probe.py"

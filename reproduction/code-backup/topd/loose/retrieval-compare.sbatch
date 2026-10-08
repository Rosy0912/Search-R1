#!/bin/bash
#SBATCH --job-name=topd-retrieval-compare
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/retrieval-compare-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
test -f "$AGENT_ROOT/logs/full-copy-verification.ready"
test -f "$AGENT_ROOT/logs/data-preparation.ready"
exec 9>"$AGENT_ROOT/gpu-job.lock"
flock -n 9
cd "$AGENT_ROOT/repo"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/retriever-gpu-lib:$AGENT_ROOT/repo"
"$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/retrieval-compare.py"

#!/bin/bash
#SBATCH --job-name=streamed-adamw-device-proof
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/streamed-adamw-device-proof-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export PYTHONPATH="$AGENT_ROOT/repo-terminal-fixed413"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$AGENT_ROOT/repo-terminal-fixed413"
exec "$AGENT_ROOT/env/bin/python" -B -m search_r1.baselines.preflight -- "$AGENT_ROOT/env/bin/python" -B "$AGENT_ROOT/tools/streamed-adamw-device-parent-v1/parent.py"

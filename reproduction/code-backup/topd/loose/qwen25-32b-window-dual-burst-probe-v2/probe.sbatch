#!/bin/bash
#SBATCH --job-name=topd-window-dual-telemetry-burst
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd-window-dual-telemetry-burst-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export PYTHONPATH="$AGENT_ROOT/repo-terminal-fixed413"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
cd "$AGENT_ROOT/repo-terminal-fixed413"
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- \
 "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/qwen25-32b-window-dual-burst-probe-v2/probe.py"

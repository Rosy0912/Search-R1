#!/bin/bash
#SBATCH --job-name=qwen25-32b-eval64
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/qwen25-32b-eval64-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export PYTHONPATH="$AGENT_ROOT/repo"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RAY_ADDRESS || true
cd "$AGENT_ROOT/repo"
test "$#" -eq 1
# No GPU context/engine until every model shard/source/tool is verified.
"$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/qwen25-32b-eval64/eval64.py" --binding "$1" --check-assets-only
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight -- \
  "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/qwen25-32b-eval64/supervisor.py" --binding "$1"

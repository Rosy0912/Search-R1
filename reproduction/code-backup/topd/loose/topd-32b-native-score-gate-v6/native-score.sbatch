#!/bin/bash
#SBATCH --job-name=topd32b-selectedteacher-numerical-release
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd32b-score-gate-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export PYTHONPATH="$AGENT_ROOT/repo-terminal-fixed413"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NVIDIA_TF32_OVERRIDE=0
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy RAY_ADDRESS || true
export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1
cd "$AGENT_ROOT/tools/topd-32b-native-score-gate-v6"
test "$#" -eq 1
# Root must install/freeze a complete tool manifest before this candidate can run.
sha256sum -c files.sha256
"$AGENT_ROOT/env/bin/python" supervisor.py --provenance "$1" --candidate "$AGENT_ROOT/tools/topd-32b-pipeline-candidate" --check-only
exec "$AGENT_ROOT/env/bin/python" -m search_r1.baselines.preflight --  "$AGENT_ROOT/env/bin/python" supervisor.py --provenance "$1" --candidate "$AGENT_ROOT/tools/topd-32b-pipeline-candidate"

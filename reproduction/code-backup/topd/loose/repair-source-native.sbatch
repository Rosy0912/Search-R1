#!/bin/bash
#SBATCH --job-name=topd-repaired-source-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/repair-source-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/repo"
git rev-parse HEAD
"$AGENT_ROOT/env/bin/python" - <<'PY'
from search_r1.baselines.topd_local_gpu import implementation_fingerprint
print('ACTUAL_GPU_SCORER_SOURCE',implementation_fingerprint(),flush=True)
PY
"$AGENT_ROOT/env/bin/python" -m pytest tests/baselines tests/reproduction -q
printf 'NATIVE_REPAIRED_SOURCE_PASS_NO_TRAINING\n'

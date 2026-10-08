#!/bin/bash
#SBATCH --job-name=topd-teacher-thread-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/teacher-thread-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$AGENT_ROOT/repo"
export PROBE_COMMIT="$(git rev-parse HEAD)"
printf 'PROBE_COMMIT=%s\n' "$PROBE_COMMIT"
"$AGENT_ROOT/env/bin/python" -m pytest tests/baselines/test_topd_runtime_audit.py -q
"$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/teacher-thread-native-probe.py"

#!/bin/bash
#SBATCH --job-name=topd-evidence-normalizer-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/repair-normalizer-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/tools"
sha256sum build_topd_execution_evidence.py test_build_topd_execution_evidence.py
"$AGENT_ROOT/env/bin/python" -m pytest test_build_topd_execution_evidence.py -q
printf 'NATIVE_NORMALIZER_FIXTURES_PASS_NO_TRAINING\n'

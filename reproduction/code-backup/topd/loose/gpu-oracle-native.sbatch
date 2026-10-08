#!/bin/bash
#SBATCH --job-name=topd-original-oracle-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/gpu-oracle-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/tools/afunix-scorer-probe"
sha256sum original_gpu_oracle.py test_original_gpu_oracle.py probe_round_checks.py test_probe_round_checks.py probe.py
"$AGENT_ROOT/env/bin/python" -m pytest test_original_gpu_oracle.py test_probe_round_checks.py -q
printf 'NATIVE_ORIGINAL_GPU_ORACLE_CPU_FIXTURES_PASS_NO_TRAINING\n'

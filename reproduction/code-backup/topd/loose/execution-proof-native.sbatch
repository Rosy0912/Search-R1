#!/bin/bash
#SBATCH --job-name=topd-execution-proof-native
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/execution-proof-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/tools"
for name in topd-short-run.py test_topd_launch_gate.py test_topd_local_gpu_ipc.py verify-topd-teacher.py verify-topd-checkpoint.py topd-five-step.sbatch build_topd_execution_evidence.py test_build_topd_execution_evidence.py; do
  sha256sum "$AGENT_ROOT/tools/$name"
done
"$AGENT_ROOT/env/bin/python" -m pytest test_topd_local_gpu_ipc.py test_topd_launch_gate.py test_topd_checkpoint.py test_verify_topd_teacher.py -q
"$AGENT_ROOT/env/bin/python" launcher-cleanup-native-probe.py
printf 'NATIVE_EXECUTION_PROOF_FIXTURES_PASS_NO_TRAINING\n'

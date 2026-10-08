#!/bin/bash
#SBATCH --job-name=topd-parallel-native-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/parallel-gpu-native-%j.log
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
"$AGENT_ROOT/env/bin/python" -m unittest discover -s tests/baselines -v
"$AGENT_ROOT/env/bin/python" -m unittest discover -s tests/reproduction -v
cd "$AGENT_ROOT/tools"
sha256sum topd-short-run.py test_topd_local_gpu_ipc.py test_topd_launch_gate.py
"$AGENT_ROOT/env/bin/python" -m pytest test_topd_local_gpu_ipc.py test_topd_launch_gate.py test_topd_checkpoint.py test_verify_topd_teacher.py -q
"$AGENT_ROOT/env/bin/python" launcher-cleanup-native-probe.py
printf 'NATIVE_PARALLEL_GPU_CPU_FIXTURES_PASS_NO_TRAINING\n'

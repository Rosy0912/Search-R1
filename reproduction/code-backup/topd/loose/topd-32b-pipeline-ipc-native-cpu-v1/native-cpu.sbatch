#!/bin/bash
#SBATCH --job-name=topd32b-ipc-cpu-v1
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd32b-ipc-native-cpu-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo-terminal-fixed413:$AGENT_ROOT/tools/topd-32b-pipeline-ipc-native-cpu-v1"
cd "$AGENT_ROOT/tools/topd-32b-pipeline-ipc-native-cpu-v1"
sha256sum topd32b_shared_pipeline/*.py test_backend.py native-cpu.sbatch
"$AGENT_ROOT/env/bin/python" - <<'PYENV'
import os,torch,tensordict,transformers
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
assert torch.__version__=='2.4.0+cu121' and tensordict.__version__=='0.5.0'
print('ACTUAL_ENV',torch.__version__,transformers.__version__,tensordict.__version__)
print('ACTUAL_PID_CGROUP',os.getpid(),Path('/proc/self/cgroup').read_text())
assert any(x.split(':',2)[-1] not in ('/','/system.slice') for x in Path('/proc/self/cgroup').read_text().splitlines())
PYENV
"$AGENT_ROOT/env/bin/python" -m pytest test_backend.py -q
printf 'NATIVE_8_UNIX_ENDPOINT_2ROUND_CPU_FAKE_SCORER_CONTRACTS_ONLY_NO_GPU_NO_TRAINING\n'

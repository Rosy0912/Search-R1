#!/bin/bash
#SBATCH --job-name=topd32b-pipeline-cpu-v1
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd32b-pipeline-native-cpu-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo-terminal-fixed413:$AGENT_ROOT/tools/topd-32b-pipeline-native-cpu-v1"
cd "$AGENT_ROOT/tools/topd-32b-pipeline-native-cpu-v1"
sha256sum pipeline.py capacity_reference.py native_gate.py test_pipeline.py native-cpu.sbatch
"$AGENT_ROOT/env/bin/python" - <<'PYENV'
import os,torch,transformers,accelerate,tensordict,inspect,hashlib
from search_r1.baselines.distillation import FrozenTeacher
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
assert torch.__version__=='2.4.0+cu121' and transformers.__version__=='4.46.3' and accelerate.__version__=='1.15.0'
method=inspect.unwrap(FrozenTeacher.log_probs)
print('ACTUAL_ENV',torch.__version__,transformers.__version__,accelerate.__version__,tensordict.__version__)
print('ACTUAL_SOURCE',inspect.getfile(method),hashlib.sha256(inspect.getsource(FrozenTeacher.log_probs).encode()).hexdigest())
assert '/repo-terminal-fixed413/' in inspect.getfile(method)
PYENV
"$AGENT_ROOT/env/bin/python" -m pytest test_pipeline.py -q
printf 'NATIVE_PIPELINE_CPU_CONTRACTS_PASS_NO_GPU_NO_TRAINING\n'

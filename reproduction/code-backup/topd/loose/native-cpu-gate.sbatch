#!/bin/bash
#SBATCH --job-name=topd-native-cpu-gate
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/native-cpu-gate-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/repo"
git rev-parse HEAD
if [ ! -f "$AGENT_ROOT/logs/installed-and-derived-verification.ready" ]; then
  "$AGENT_ROOT/env/bin/python" "$AGENT_ROOT/tools/verify-installed-and-derived.py"
fi
"$AGENT_ROOT/env/bin/python" - <<'PY'
import tensordict
from tensordict import TensorDict
assert tensordict.__version__ == '0.5.0'
assert TensorDict({},batch_size=[1]).get('step_ids',None) is None
print('NATIVE_TENSORDICT_05_OPTIONAL_ACCESS_PASS',tensordict.__file__,flush=True)
PY
"$AGENT_ROOT/env/bin/python" -m pytest tests/baselines tests/reproduction -q
"$AGENT_ROOT/env/bin/python" -m unittest discover -s tests/baselines -v
"$AGENT_ROOT/env/bin/python" -m unittest discover -s tests/reproduction -v
printf 'NATIVE_CPU_GATE_PASS\n'

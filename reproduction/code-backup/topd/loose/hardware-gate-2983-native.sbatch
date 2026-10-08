#!/bin/bash
#SBATCH --job-name=topd-readonly-hardware-gate
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/hardware-gate-2983-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/repo"
cd "$AGENT_ROOT"
"$AGENT_ROOT/env/bin/python" - <<'PY'
from pathlib import Path
import hashlib,importlib.util,json
import torch
root=Path('/dataset/shaoshitong/search-r1-rl/topd')
source=root/'tools/topd-short-run.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'
binding_path=root/'logs/topd-hardware-gate-binding-2983.json'
assert hashlib.sha256(binding_path.read_bytes()).hexdigest()=='4c9aee4cc30a7380015e7decc0ee896d4f2e39d8d4c59293a941b05c83b35274'
binding=json.loads(binding_path.read_text())
assert binding['scope']=='READ_ONLY_HARDWARE_GATE_ONLY_NOT_TRAINING_BINDING'
assert binding['execution_proof']['sha256']=='6c0f3dd62acbead63294a5dd5a8d1ea7a130c3ec3518c058223fa8d5ce687382'
assert not torch.cuda.is_initialized()
spec=importlib.util.spec_from_file_location('topd_readonly_hardware_launcher',source)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
launcher.execution_proof_gate(binding)
assert not torch.cuda.is_initialized()
print('ACTUAL_READ_ONLY_HARDWARE_EXECUTION_PROOF_GATE_PASS_NO_TRAINING_NO_TEACHER_APPROVAL')
PY

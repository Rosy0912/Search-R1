#!/bin/bash
#SBATCH --job-name=topd-normalize-hardware2983
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/normalize-hardware-2983-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/repo"
cd "$AGENT_ROOT"
"$AGENT_ROOT/env/bin/python" - <<'PY'
from pathlib import Path
import hashlib
expected={'build_topd_execution_evidence.py':'01d2232173853b1967dcc9124635b371aa7dd42ae961c6badcc4dda2f85cb1b7','topd-short-run.py':'d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'}
for name,h in expected.items():assert hashlib.sha256((Path('tools')/name).read_bytes()).hexdigest()==h,name
assert hashlib.sha256(Path('tools/afunix-scorer-probe/probe.py').read_bytes()).hexdigest()=='156747b748e5c82451246edbfb82923331a205bb4d33254c398f0f4206c03d38'
PY
"$AGENT_ROOT/env/bin/python" tools/build_topd_execution_evidence.py \
 --root "$AGENT_ROOT" --probe-dir "$AGENT_ROOT/logs/afunix-scorer-hardware-2983" \
 --probe-script "$AGENT_ROOT/tools/afunix-scorer-probe/probe.py" \
 --output "$AGENT_ROOT/logs/hardware-evidence-2983.json"

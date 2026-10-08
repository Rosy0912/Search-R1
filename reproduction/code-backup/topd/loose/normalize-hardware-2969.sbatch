#!/bin/bash
#SBATCH --job-name=topd-normalize-hardware2969
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/normalize-hardware-2969-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/repo"
cd "$AGENT_ROOT"
"$AGENT_ROOT/env/bin/python" - <<'PY'
from pathlib import Path
import hashlib
p=Path('tools/build_topd_execution_evidence.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1a7c17ba547381d95f4c7dd6133f2f39b99dcebba233efa38251b17590479d83'
PY
"$AGENT_ROOT/env/bin/python" tools/build_topd_execution_evidence.py \
 --root "$AGENT_ROOT" --probe-dir "$AGENT_ROOT/logs/afunix-scorer-hardware-2969" \
 --probe-script "$AGENT_ROOT/tools/afunix-scorer-probe/probe.py" \
 --output "$AGENT_ROOT/logs/hardware-evidence-2969.json"

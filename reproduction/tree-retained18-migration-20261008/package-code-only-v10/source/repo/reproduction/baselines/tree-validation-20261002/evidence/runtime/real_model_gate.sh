#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
source "$ROOT/env/bin/activate"
cd "$ROOT/repo"
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NODELIST:-}" = h100-9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 VLLM_NO_USAGE_STATS=1 VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [ "${SEARCH_R1_GPU_PREFLIGHT:-0}" != 1 ]; then
 exec python -m search_r1.baselines.preflight -- bash "$ROOT/tools/real_model_gate.sh"
fi
python - <<'PY'
import json,os
from types import SimpleNamespace
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus,validate_gpu_lease
validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=8)))
i,p=gpu_snapshot();r=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),i,p)
assert len(r)==8 and {x['uuid'] for x in r}==set(json.load(open('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/expected-gpu-uuids.json')))
PY
python "$ROOT/tools/real_model_probe.py" > "$ROOT/logs/model-smoke-20261003.log" 2>&1

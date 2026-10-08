#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
source "$ROOT/env/bin/activate"
cd "$ROOT/repo"
test -f "$ROOT/logs/environment.ready"
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_NODELIST:-}" = h100-9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [ "${SEARCH_R1_GPU_PREFLIGHT:-0}" != 1 ]; then
 exec python -m search_r1.baselines.preflight -- bash "$ROOT/tools/gpu_hardware_gate.sh"
fi
python - <<'PY'
import json,os
from types import SimpleNamespace
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus,validate_gpu_lease
validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=8)))
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
i,p=gpu_snapshot();r=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),i,p)
assert len(r)==8 and {x['uuid'] for x in r}==set(json.load(open(root+'/tools/expected-gpu-uuids.json')))
PY
PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
python -m torch.distributed.run --master_addr=127.0.0.1 --master_port="$PORT" --nproc_per_node=8 "$ROOT/repo/reproduction/h20/cuda_preflight.py" > "$ROOT/logs/hardware-cuda.log" 2>&1
python "$ROOT/tools/gpu_hardware_probe.py" > "$ROOT/logs/hardware-flash.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/cosearch
export PATH="$AGENT_ROOT/env/bin:$PATH"
export TMPDIR="$AGENT_ROOT/tmp" HF_HOME="$AGENT_ROOT/cache/hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export NCCL_NVLS_ENABLE=0 VLLM_ATTENTION_BACKEND=XFORMERS OMP_NUM_THREADS=1
cd "$AGENT_ROOT/repo"
test -f "$AGENT_ROOT/logs/environment.ready"
test -n "${SLURM_JOB_ID:?Run inside our exclusive Slurm allocation}"
python - <<'PY'
import json, os
from search_r1.baselines.preflight import gpu_snapshot, check_selected_gpus
selected = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(selected) == 8, selected
inventory, processes = gpu_snapshot()
print('HARDWARE_SCHEDULER_SNAPSHOT', json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
    visible=selected, selected=check_selected_gpus(selected, inventory, processes),
    existing_process_gpu_uuids=sorted(processes)), sort_keys=True), flush=True)
PY
MASTER_PORT=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
python -m torch.distributed.run --standalone --nnodes=1 --master_port="$MASTER_PORT" \
  --nproc_per_node=8 reproduction/h20/cuda_preflight.py
python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1,128,8,64,device='cuda',dtype=torch.bfloat16)
result = flash_attn_func(q,q,q,causal=True)
assert result.shape == q.shape and torch.isfinite(result).all()
print('FLASH_ATTENTION_BF16_OK', flush=True)
PY
FIRST_ALLOCATED=${CUDA_VISIBLE_DEVICES%%,*}
CUDA_VISIBLE_DEVICES="$FIRST_ALLOCATED" python "$AGENT_ROOT/tools/model_hardware_smoke.py"
touch "$AGENT_ROOT/logs/hardware.ready"

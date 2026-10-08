#!/bin/bash
set -euo pipefail
export AGENT_ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
ROOT="$AGENT_ROOT"
test "$SLURM_JOB_NODELIST" = h100-8
test "$(hostname -s)" = h100-8
test -f "$ROOT/logs/full-copy-verification.ready"
test -f "$ROOT/logs/environment.ready"
exec 9>"$ROOT/gpu-job.lock"
flock -n 9
export PATH="$ROOT/env/bin:$PATH" PYTHONPATH="$ROOT/repo"
export C_INCLUDE_PATH="$ROOT/python-dev/usr/include/python3.10:$ROOT/python-dev/usr/include:$ROOT/python-dev/usr/include/x86_64-linux-gnu/python3.10"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_ATTENTION_BACKEND=XFORMERS
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$ROOT/repo"
RUN="$ROOT/runs/hardware-$SLURM_JOB_ID"
mkdir -p "$RUN"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
scontrol show job "$SLURM_JOB_ID" > "$RUN/slurm-job.txt"
python - <<'PY'
import json,os,pathlib
from search_r1.baselines.preflight import gpu_snapshot,check_selected_gpus
inventory,processes=gpu_snapshot()
selected=check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
assert len(selected)==8
pathlib.Path(os.environ['AGENT_ROOT']+'/tools/expected-gpu-uuids.json').write_text(json.dumps([row['uuid'] for row in selected]))
PY
python "$ROOT/tools/resource_guard.py" --output "$RUN/resource-samples.jsonl" &
GUARD_PID=$!
for attempt in $(seq 1 30); do
 kill -0 "$GUARD_PID"
 test ! -f "$RUN/resource-samples.ready" || break
 sleep 1
done
test -f "$RUN/resource-samples.ready"
trap 'kill -TERM "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true' EXIT
python -m search_r1.baselines.preflight -- python -m torch.distributed.run --nproc_per_node=8 --master_port=26381 reproduction/h20/cuda_preflight.py
sleep 3
python -m search_r1.baselines.preflight -- python "$ROOT/tools/check_actor_compile.py" --output "$RUN/actor-entropy-compile.json"
sleep 3
python -m search_r1.baselines.preflight -- python "$ROOT/tools/general-model-preflight.py"
sleep 3
kill -0 "$GUARD_PID"
export HF_DATASETS_CACHE="$ROOT/cache/retriever-datasets"
PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" python -m search_r1.baselines.preflight -- python "$ROOT/tools/reference-retrieval-compare.py"
kill -0 "$GUARD_PID"
git rev-parse HEAD > "$RUN/source-commit.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$RUN/gpu-after.txt"
echo REAL_NCCL_FLASH_3B_AND_LOCAL_RETRIEVAL_GATE_FINISHED
printf '0\n' > "$ROOT/logs/hardware-$SLURM_JOB_ID.exit"

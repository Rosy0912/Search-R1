#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?explicit common8 base required}"
COMMIT="${SEARCH_G1_SOURCE_COMMIT:?committed snapshot required}"
: "${SLURM_JOB_ID:?native allocation required}"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export SEARCH_G1_PROFILE=common8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" XDG_CACHE_HOME="$BASE/cache" HF_HOME="$BASE/cache/hf"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS
mkdir -p "$BASE/logs" "$BASE/tmp" "$BASE/ray" "$BASE/artifacts" "$BASE/cache/triton" "$BASE/cache/inductor" "$BASE/cache/hf"
cd "$BASE/repo"
# Model/data preparation is inside the actual native full-memory allocation.
python - <<'PY'
import os,socket,subprocess
from reproduction.common8.native_allocation import fields
from reproduction.common8.stage_retrieval import memory_bytes
job=fields(subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True))
assert job['JobState']=='RUNNING' and job['NodeList']==socket.gethostname().split('.')[0]=='h100-8'
assert job['JobId']==os.environ['SLURM_JOB_ID'] and job['NumNodes']=='1' and job['TresPerNode'] in ('gres:gpu:8','gres:gpu:h100:8')
assert memory_bytes(job['TRES'])>=512*2**30
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
PY
prepare_options=()
if [[ "${SEARCH_G1_ALLOW_EXISTING_GPU_PROCESSES:-0}" == 1 ]]; then
  prepare_options+=(--allow-existing-gpu-processes)
fi
prepare_options+=(--rollout-memory-utilization "${SEARCH_G1_ROLLOUT_MEMORY_UTILIZATION:-0.4}")
python -m reproduction.common8.prepare_run "${prepare_options[@]}" --base "$BASE" \
  --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725" \
  --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit "$COMMIT"
exec python -m reproduction.common8.run_pipeline --base "$BASE" --export-port "${SEARCH_G1_EXPORT_PORT:-19874}"

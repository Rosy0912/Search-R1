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
if [[ -n "${SEARCH_G1_SLURM_SHARED_AUTHORIZATION:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
# Model/data preparation is inside the actual native full-memory allocation.
python - <<'PY'
import os,socket,subprocess
from reproduction.common8.native_allocation import fields
from reproduction.common8.stage_retrieval import memory_bytes
job=fields(subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True))
assert job['JobState']=='RUNNING' and job['NodeList']==socket.gethostname().split('.')[0]=='h100-8'
assert job['JobId']==os.environ['SLURM_JOB_ID'] and job['NumNodes']=='1'
if os.environ.get('SEARCH_G1_SLURM_SHARED_AUTHORIZATION'):
    from reproduction.common8.shared_slurm import authorization
    authorization(os.environ['SEARCH_G1_SLURM_SHARED_AUTHORIZATION'],os.environ['SEARCH_G1_BASE'])
    assert job.get('TresPerNode','(null)') in ('(null)','N/A') and job['OverSubscribe']!='NO'
else:
    assert job['TresPerNode'] in ('gres:gpu:8','gres:gpu:h100:8')
assert memory_bytes(job['TRES'])>=512*2**30
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
PY
prepare_options=()
if [[ "${SEARCH_G1_ALLOW_EXISTING_GPU_PROCESSES:-0}" == 1 ]]; then
  prepare_options+=(--allow-existing-gpu-processes)
fi
if [[ -n "${SEARCH_G1_LOCAL_RETRIEVAL_ASSETS:-}" ]]; then
  prepare_options+=(--local-retrieval-assets "$SEARCH_G1_LOCAL_RETRIEVAL_ASSETS")
fi
prepare_options+=(--rollout-memory-utilization "${SEARCH_G1_ROLLOUT_MEMORY_UTILIZATION:-0.1}")
if [[ -n "${SEARCH_G1_FULL_STATE_RESTORE_RECEIPT:-}" ]]; then
  prepare_options+=(--restore-receipt "$SEARCH_G1_FULL_STATE_RESTORE_RECEIPT")
fi
python -m reproduction.common8.prepare_run "${prepare_options[@]}" --base "$BASE" \
  --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725" \
  --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit "$COMMIT"
exec python -m reproduction.common8.run_pipeline --base "$BASE" --export-port "${SEARCH_G1_EXPORT_PORT:-19874}"

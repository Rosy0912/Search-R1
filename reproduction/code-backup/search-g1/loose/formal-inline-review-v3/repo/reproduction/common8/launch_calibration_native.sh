#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?explicit calibration child base required}"
PARENT="${SEARCH_G1_PRODUCTION_BASE:?distinct fresh production parent required}"
ROLE="${SEARCH_G1_CALIBRATION_ROLE:?explicit discovery or confirmation role required}"
COMMIT="${SEARCH_G1_SOURCE_COMMIT:?committed snapshot required}"
: "${SLURM_JOB_ID:?native allocation required}"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export SEARCH_G1_PROFILE=common8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" XDG_CACHE_HOME="$BASE/cache" HF_HOME="$BASE/cache/hf"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS
cd "$BASE/repo"
args=(--base "$BASE" --parent-base "$PARENT" --role "$ROLE"
  --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725"
  --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit "$COMMIT")
if [[ "$ROLE" == policy_safety_calibration_confirmation ]]; then
  args+=(--discovery-base "${SEARCH_G1_DISCOVERY_BASE:?completed discovery base required}")
fi
# This validates the actual eight-H100/512GiB reservation before reading the
# official model/data. It does not synthesize GPU or healthy admission markers.
python -m reproduction.common8.prepare_calibration_run "${args[@]}"
exec python -m reproduction.common8.run_pipeline --base "$BASE" --export-port "${SEARCH_G1_EXPORT_PORT:-19874}"

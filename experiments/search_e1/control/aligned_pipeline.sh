#!/usr/bin/env bash
set -euo pipefail
BASE=${E1_BASE:?};PROFILE=r1;export E1_PROFILE=$PROFILE
CONTROL="$BASE/source/experiments/search_e1/control"
PYTHON="$BASE/env/bin/python"
test -f "$BASE/receipts/aligned-config-approved.json"
E1_ALIGNED_STEPS=$("$PYTHON" -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["E1_BASE"])/"receipts/aligned-config-approved.json").read_text())["grpo_updates_per_stage"])')
export E1_ALIGNED_STEPS
[[ "$E1_ALIGNED_STEPS" == 200 || "$E1_ALIGNED_STEPS" == 300 ]] || exit 2
export E1_GPUS=8 E1_VIRTUAL_DP_WORLD=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export E1_PHYSICAL_MICRO=1 E1_PHYSICAL_SCORE=1 E1_JIT_OPTIMIZER=1 E1_LOGIT_CHUNK=128 E1_OPTIMIZER_CHUNK=1 E1_STAGE_CHUNK_STATES=1
export E1_VLLM_UTIL=.25 E1_INFERENCE_UTIL=.85 E1_RETRIEVER_URL=http://10.120.17.101:18001/retrieve
export HF_HOME="$BASE/hf-cache" HF_DATASETS_CACHE="$BASE/hf-datasets" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RAY_enable_metrics_collection=0 WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
unset E1_FULL_RESUME_STATE E1_MODEL
exec 9>"$BASE/experiment.lock"
flock -n 9 || exit 2
phase() { printf '%s %s\n' "$(date -Is)" "$1" > "$BASE/receipts/pipeline-stage.txt"; }
trap 'code=$?; phase "EXIT_$code"' EXIT
phase VERIFYING_FROZEN_RUNTIME_AND_RETRIEVER
"$PYTHON" "$CONTROL/verify_runtime.py"
E1_RUNTIME_PURPOSE=opsd "$PYTHON" "$CONTROL/verify_runtime.py"
RUNTIME="$BASE/runtime/r1";OPSD_RUNTIME="$BASE/opsd-runtime/r1"
export PYTHONPATH="$RUNTIME/upstream/Search-R1:$RUNTIME"
OPSD_ACCUMULATION=4;GPUS=8;steps=$E1_ALIGNED_STEPS
"$PYTHON" - <<'PY'
import urllib.request,json
health=json.load(urllib.request.urlopen('http://10.120.17.101:18001/health',timeout=20))
assert health['ready'] and health['ntotal']==21015324 and health['tf32'] is False
assert health['acceptance_sha256']=='f256ee31dea1297a07baa28ca9b722e678fbc3555669a3d2bcebcb00884a9443'
PY
# No stopped two-iteration test: one fresh full run saves extra2/5 evidence.
source "$CONTROL/aligned_pipeline_stages.sh"

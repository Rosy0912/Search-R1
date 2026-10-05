#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?common8 base must be explicit}"
export SEARCH_G1_PROFILE=common8 RAY_DEDUP_LOGS=0
export SEARCH_G1_BASE="$BASE"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
unset PYTHONHOME RAY_ADDRESS
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
export VLLM_ATTENTION_BACKEND=XFORMERS PYTHONUNBUFFERED=1
export PROBE_URL=http://127.0.0.1:18012/judge_probe
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=probe ECO_PROBE_GATE=1
export ECO_BAL_GROUND=0.6 ECO_BAL_NOSEARCH_BONUS=0.5 ECO_W_COST=0.02
export ECO_WRONG_GROUND_SCALE=0 ECO_PROBE_BASE=0 ECO_PROBE_ALPHA_COST=0 ECO_USE_CF=0
export ECO_NO_SELF=1 ECO_PROBE_MAX_WAIT=300
cd "$BASE/repo"
python -m reproduction.common8.probe_admission --verify
python -m reproduction.common8.claim_training
# Fresh official actor/reference and the same resolved recipe used by model admission.
python -m reproduction.common8.configuration 2>&1 | tee -a "$SEARCH_G1_TRAIN_LOG"

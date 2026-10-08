#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?fresh A800 base must be explicit}"
export SEARCH_G1_PROFILE=common8 SEARCH_G1_TRAINING_WORLD_SIZE=4 RAY_DEDUP_LOGS=0
export SEARCH_G1_BASE="$BASE"
export PATH="/data/search-g1/env/bin:$PATH" PYTHONPATH="$BASE/repo"
unset PYTHONHOME RAY_ADDRESS
export C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include'
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export RAY_TMPDIR="${RAY_TMPDIR:?bound private Ray root must be inherited from the pipeline}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
export VLLM_ATTENTION_BACKEND=XFORMERS PYTHONUNBUFFERED=1
export PROBE_URL=http://127.0.0.1:18012/judge_probe
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=probe ECO_PROBE_GATE=1
export ECO_BAL_GROUND=0.6 ECO_BAL_NOSEARCH_BONUS=0.5 ECO_W_COST=0.02
export ECO_WRONG_GROUND_SCALE=0 ECO_PROBE_BASE=0 ECO_PROBE_ALPHA_COST=0 ECO_USE_CF=0
export ECO_NO_SELF=1 ECO_PROBE_MAX_WAIT=300
cd "$BASE/repo"
python -c 'import json,os; from pathlib import Path; from reproduction.common8.formal_contract import assert_formal_run; from reproduction.common8.run_pipeline import supported_execution_role; from reproduction.common8.topology import execution_job_id,world_size; base=Path(os.environ["SEARCH_G1_BASE"]); run=json.loads((base/"logs/run-manifest.json").read_text()); assert_formal_run(run); assert run["execution_backend"]=="ssh_local" and execution_job_id(run) is None and world_size(run)==4; assert supported_execution_role(run,base)=="production"; assert os.environ["CUDA_VISIBLE_DEVICES"]=="0,1,2,3"'
python -m reproduction.common8.probe_admission --verify
# The real fresh0 probes, local allocation, immutable model and bound source
# are frozen prospectively; the actual full2560 behavior gate runs before Adam.
python -m reproduction.common8.formal_inline_safety --prepare --base "$BASE"
python -m reproduction.common8.claim_training
# The shared trainer writes checkpoint5, audits it inline, and continues to200
# in this same training process when the real acceptance succeeds.
python -m reproduction.common8.configuration 2>&1 | tee -a "$SEARCH_G1_TRAIN_LOG"

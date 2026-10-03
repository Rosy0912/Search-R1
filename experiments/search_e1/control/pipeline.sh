#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?};PROFILE=${E1_PROFILE:?};CONTROL="$BASE/source/experiments/search_e1/control"
PYTHON="$BASE/env/bin/python"
mkdir -p "$BASE/receipts" "$BASE/logs" "$BASE/runs" "$BASE/tmp" "$BASE/hf-datasets" "$BASE/cache"
export HF_HOME="$BASE/hf-cache" HF_DATASETS_CACHE="$BASE/hf-datasets" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec 9>"$BASE/experiment.lock"
flock -n 9 || { echo 'This task pipeline already has a supervisor'; exit 2; }
phase() { printf '%s %s\n' "$(date -Is)" "$1" > "$BASE/receipts/pipeline-stage.txt"; }
trap 'code=$?; phase "EXIT_$code"' EXIT
phase WAITING_FOR_ENVIRONMENT_AND_ASSETS
for ((n=0;n<4320;n++)); do
 if [[ -f "$BASE/receipts/env.ready" ]]; then
  if [[ "$PROFILE" == archive && -f "$BASE/receipts/assets.ready" ]] || [[ "$PROFILE" == r1 && -f "$BASE/receipts/local-assets.ready" ]]; then break; fi
 fi
 sleep 10
done
test -f "$BASE/receipts/env.ready"
if [[ "$PROFILE" == archive ]]; then test -f "$BASE/receipts/assets.ready"; else test -f "$BASE/receipts/local-assets.ready"; fi
phase BUILDING_FROZEN_RUNTIME
"$PYTHON" "$CONTROL/build_runtime.py" --base "$BASE" --profile "$PROFILE"
RUNTIME="$BASE/runtime/$PROFILE"
export PYTHONPATH="$RUNTIME/upstream/Search-R1:$RUNTIME" WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export E1_DATASET_ROOT="$BASE/assets/datasets"
"$PYTHON" -m search_e1.prepare --output "$BASE/data" > "$BASE/logs/prepare.log" 2>&1
"$PYTHON" "$CONTROL/prepare_common_eval.py"
E1_TEST_RUNTIME="$RUNTIME" E1_TEST_ARCHIVE="$BASE/source/experiments/search_e1/archive" "$PYTHON" -m unittest discover -s "$BASE/source/experiments/search_e1/tests" -p test_overlay.py -v > "$BASE/receipts/overlay-tests.txt" 2>&1
if [[ "$PROFILE" == archive ]]; then
 phase STARTING_LOCAL_RETRIEVER
 "$PYTHON" "$CONTROL/retriever_cpu.py" > "$BASE/logs/retriever.log" 2>&1 &
 retriever_pid=$!
 printf '%s\n' "$retriever_pid" > "$BASE/receipts/retriever.pid"
 # Keep the task retrieval service alive for B even after A finishes; do not trap-kill it.
fi
phase WAITING_FOR_RETRIEVER
"$PYTHON" - <<'PY'
import requests,time,os,json
from pathlib import Path
base=Path(os.environ['E1_BASE']);url='http://10.123.4.11:18000/retrieve'
for _ in range(1800):
 try:
  r=requests.post(url,json={'queries':['capital of France'],'topk':3,'return_scores':True},timeout=30);r.raise_for_status();x=r.json();assert len(x['result'][0])==3
  (base/'receipts/retriever-query.json').write_text(json.dumps(x,indent=2));break
 except (requests.RequestException,AssertionError):time.sleep(2)
else:raise SystemExit('Retriever not ready; no training started')
PY
phase GPU_REAL_UPDATE_GATE_AND_MEMORY_TUNING
"$PYTHON" "$CONTROL/tune_grpo_memory.py"
export E1_VLLM_UTIL=$(cat "$BASE/receipts/$PROFILE-selected-util.txt")
export E1_INFERENCE_UTIL=0.85
phase FRESH_GRPO_ROUND_1
export E1_STAGE=round-1-grpo
"$PYTHON" "$CONTROL/run_checked.py"
if [[ "$PROFILE" == archive ]]; then steps=200; else steps=500; fi
checkpoint="$BASE/runs/$PROFILE/round-1-grpo/actor/global_step_$steps"
test -f "$checkpoint/config.json"
"$PYTHON" "$CONTROL/verify_checkpoint.py" --initial "$BASE/assets/model" --checkpoint "$checkpoint" --report "$BASE/receipts/$PROFILE-round1-checkpoint.json"
phase INDEPENDENT_ROUND_1_EVALUATION
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/independent_round_1_evaluation-admission.json"
eval_steps=(200)
if [[ "$PROFILE" == r1 ]]; then eval_steps+=(500); fi
for eval_step in "${eval_steps[@]}"; do
 "$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/eval-$eval_step-admission.json"
 "$PYTHON" -m search_e1.rollout --model "$BASE/runs/$PROFILE/round-1-grpo/actor/global_step_$eval_step" --data "$BASE/data/test-common-64.jsonl" --output "$BASE/runs/$PROFILE/round1-step-$eval_step-test64.jsonl" --k 1 --backend vllm --workers 4 --temperature 0 --retriever-url http://10.123.4.11:18000/retrieve > "$BASE/logs/round1-step-$eval_step-test64.log" 2>&1
done
phase OPSD_ROUND_1_TRAJECTORIES
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_1_trajectories-admission.json"
"$PYTHON" -m search_e1.rollout --model "$checkpoint" --data "$BASE/data/train.jsonl" --output "$BASE/runs/$PROFILE/round1-rollouts.jsonl" --k 5 --backend vllm --workers 4 --retriever-url http://10.123.4.11:18000/retrieve > "$BASE/logs/round1-rollouts.log" 2>&1
"$PYTHON" -m search_e1.mine --rollouts "$BASE/runs/$PROFILE/round1-rollouts.jsonl" --output "$BASE/runs/$PROFILE/round1-pairs.jsonl"
phase OPSD_ROUND_1
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_1-admission.json"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m search_e1.opsd --model "$checkpoint" --pairs "$BASE/runs/$PROFILE/round1-pairs.jsonl" --validation "$BASE/data/validation.jsonl" --accumulation 8 --output "$BASE/runs/$PROFILE/round1-opsd" --retriever-url http://10.123.4.11:18000/retrieve > "$BASE/logs/round1-opsd.log" 2>&1
export E1_MODEL="$BASE/runs/$PROFILE/round1-opsd/merged"
test -f "$E1_MODEL/config.json"
phase FRESH_GRPO_ROUND_2
export E1_STAGE=round-2-grpo
"$PYTHON" "$CONTROL/run_checked.py"
checkpoint="$BASE/runs/$PROFILE/round-2-grpo/actor/global_step_$steps"
test -f "$checkpoint/config.json"
"$PYTHON" "$CONTROL/verify_checkpoint.py" --initial "$E1_MODEL" --checkpoint "$checkpoint" --report "$BASE/receipts/$PROFILE-round2-checkpoint.json"
phase OPSD_ROUND_2_TRAJECTORIES
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_2_trajectories-admission.json"
"$PYTHON" -m search_e1.rollout --model "$checkpoint" --data "$BASE/data/train.jsonl" --output "$BASE/runs/$PROFILE/round2-rollouts.jsonl" --k 5 --backend vllm --workers 4 --retriever-url http://10.123.4.11:18000/retrieve > "$BASE/logs/round2-rollouts.log" 2>&1
"$PYTHON" -m search_e1.mine --rollouts "$BASE/runs/$PROFILE/round2-rollouts.jsonl" --output "$BASE/runs/$PROFILE/round2-pairs.jsonl"
phase OPSD_ROUND_2
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_2-admission.json"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m search_e1.opsd --model "$checkpoint" --pairs "$BASE/runs/$PROFILE/round2-pairs.jsonl" --validation "$BASE/data/validation.jsonl" --accumulation 8 --output "$BASE/runs/$PROFILE/round2-opsd" --retriever-url http://10.123.4.11:18000/retrieve > "$BASE/logs/round2-opsd.log" 2>&1
phase FINAL_SEVEN_BENCHMARK_EVALUATION
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/final_seven_benchmark_evaluation-admission.json"
"$PYTHON" -m search_e1.rollout --model "$BASE/runs/$PROFILE/round2-opsd/merged" --data "$BASE/data/test.jsonl" --output "$BASE/runs/$PROFILE/final-test.jsonl" --k 1 --backend vllm --workers 4 --temperature 0 --retriever-url http://10.123.4.11:18000/retrieve > "$BASE/logs/final-test.log" 2>&1
phase COMPLETED_ALL_STAGES

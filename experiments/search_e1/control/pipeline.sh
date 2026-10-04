#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?};PROFILE=${E1_PROFILE:?};CONTROL="$BASE/source/experiments/search_e1/control"
PYTHON="$BASE/env/bin/python"
GPUS=${E1_GPUS:-4}
[[ "$GPUS" == 4 || "$GPUS" == 8 ]] || exit 2
OPSD_ACCUMULATION=$((32 / GPUS))
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
"$PYTHON" -c 'from ray.dashboard.modules.reporter.reporter_agent import ReporterAgent; from ray.dashboard import optional_deps; print("RAY_REPORTER_READY")'
phase BUILDING_FROZEN_RUNTIME
if [[ "${E1_REUSE_RUNTIME:-0}" == 1 ]]; then
 "$PYTHON" "$CONTROL/verify_runtime.py"
else
 "$PYTHON" "$CONTROL/build_runtime.py" --base "$BASE" --profile "$PROFILE"
fi
RUNTIME="$BASE/runtime/$PROFILE"
export PYTHONPATH="$RUNTIME/upstream/Search-R1:$RUNTIME" WANDB_MODE=offline PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export E1_DATASET_ROOT="$BASE/assets/datasets"
"$PYTHON" -m search_e1.prepare --output "$BASE/data" > "$BASE/logs/prepare.log" 2>&1
"$PYTHON" "$CONTROL/prepare_common_eval.py"
E1_TEST_RUNTIME="$RUNTIME" E1_TEST_ARCHIVE="$BASE/source/experiments/search_e1/archive" "$PYTHON" -m unittest discover -s "$BASE/source/experiments/search_e1/tests" -p test_overlay.py -v > "$BASE/receipts/overlay-tests.txt" 2>&1
if [[ "$PROFILE" == archive ]]; then
 phase STARTING_LOCAL_RETRIEVER
 # An independent tmux session survives supervisor exit and serves B.
 if ! tmux has-session -t e1-retriever-20261004 2>/dev/null; then
  printf -v retriever_command 'E1_BASE=%q CUDA_VISIBLE_DEVICES=%q bash %q > %q 2>&1' "$BASE" "$CUDA_VISIBLE_DEVICES" "$CONTROL/retriever_service.sh" "$BASE/logs/retriever.log"
  tmux new-session -d -s e1-retriever-20261004 "$retriever_command"
 fi
fi
phase WAITING_FOR_RETRIEVER
"$PYTHON" - <<'PY'
import requests,time,os,json
from pathlib import Path
base=Path(os.environ['E1_BASE']);url='http://10.120.17.101:18001/retrieve'
for _ in range(1800):
 try:
  h=requests.get('http://10.120.17.101:18001/health',timeout=10);h.raise_for_status();backend=h.json()
  assert backend['ready'] and backend['ntotal']==21015324 and backend['tf32'] is False and backend['serialized']
  (base/'receipts/retriever-backend.json').write_text(json.dumps(backend,indent=2))
  r=requests.post(url,json={'queries':['capital of France'],'topk':3,'return_scores':True},timeout=120);r.raise_for_status();x=r.json();assert len(x['result'][0])==3
  (base/'receipts/retriever-query.json').write_text(json.dumps(x,indent=2));break
 except (requests.RequestException,AssertionError):time.sleep(2)
else:raise SystemExit('Retriever not ready; no training started')
PY
phase GPU_REAL_UPDATE_GATE_AND_MEMORY_TUNING
E1_RESUME_GATES="${E1_RESUME_GATES:-0}" "$PYTHON" "$CONTROL/tune_grpo_memory.py"
export E1_VLLM_UTIL=$(cat "$BASE/receipts/$PROFILE-selected-util.txt")
export E1_INFERENCE_UTIL=0.85
phase BUILDING_FROZEN_OPSD_RUNTIME
OPSD_RUNTIME="$BASE/opsd-runtime/$PROFILE"
if [[ -d "$OPSD_RUNTIME" ]]; then
 E1_RUNTIME_PURPOSE=opsd "$PYTHON" "$CONTROL/verify_runtime.py"
else
 "$PYTHON" "$CONTROL/build_runtime.py" --base "$BASE" --profile "$PROFILE" --purpose opsd
fi
phase OPSD_REAL_MEMORY_UPDATE_GATE
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd-memory-gate-admission.json"
PYTHONPATH="$OPSD_RUNTIME/upstream/Search-R1:$OPSD_RUNTIME" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" "$CONTROL/gpu_opsd_gate.py" > "$BASE/logs/opsd-memory-gate.log" 2>&1
# The heldout selection may choose the initial model; validate the final trained
# checkpoint separately so weight-change evidence is not confused with selection.
select_grpo() {
 local round=$1 initial=$2 stage="$BASE/runs/$PROFILE/round-$1-grpo"
 "$PYTHON" "$CONTROL/select_grpo_checkpoint.py" --log "$stage/train.log" --validation "$BASE/data/validation.jsonl" --initial "$initial" --stage "$stage" --steps "$steps" --report "$BASE/receipts/$PROFILE-round$round-selection.json"
}
phase FRESH_GRPO_ROUND_1
export E1_STAGE=round-1-grpo
"$PYTHON" "$CONTROL/run_checked.py"
if [[ "$PROFILE" == archive ]]; then steps=200; else steps=500; fi
checkpoint="$BASE/runs/$PROFILE/round-1-grpo/actor/global_step_$steps"
test -f "$checkpoint/config.json"
"$PYTHON" "$CONTROL/verify_checkpoint.py" --initial "$BASE/assets/model" --checkpoint "$checkpoint" --report "$BASE/receipts/$PROFILE-round1-checkpoint.json"
checkpoint=$(select_grpo 1 "$BASE/assets/model")
phase INDEPENDENT_ROUND_1_EVALUATION
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/independent_round_1_evaluation-admission.json"
eval_steps=(200)
if [[ "$PROFILE" == r1 ]]; then eval_steps+=(500); fi
for eval_step in "${eval_steps[@]}"; do
 "$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/eval-$eval_step-admission.json"
 "$PYTHON" -m search_e1.rollout --model "$BASE/runs/$PROFILE/round-1-grpo/actor/global_step_$eval_step" --data "$BASE/data/test-common-64.jsonl" --output "$BASE/runs/$PROFILE/round1-step-$eval_step-test64.jsonl" --k 1 --backend vllm --workers "$GPUS" --temperature 0 --retriever-url http://10.120.17.101:18001/retrieve > "$BASE/logs/round1-step-$eval_step-test64.log" 2>&1
done
phase OPSD_ROUND_1_TRAJECTORIES
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_1_trajectories-admission.json"
"$PYTHON" -m search_e1.rollout --model "$checkpoint" --data "$BASE/data/train.jsonl" --output "$BASE/runs/$PROFILE/round1-rollouts.jsonl" --k 5 --backend vllm --workers "$GPUS" --retriever-url http://10.120.17.101:18001/retrieve > "$BASE/logs/round1-rollouts.log" 2>&1
"$PYTHON" -m search_e1.mine --rollouts "$BASE/runs/$PROFILE/round1-rollouts.jsonl" --output "$BASE/runs/$PROFILE/round1-pairs.jsonl"
phase OPSD_ROUND_1
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_1-admission.json"
PYTHONPATH="$OPSD_RUNTIME/upstream/Search-R1:$OPSD_RUNTIME" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" -m search_e1.opsd --model "$checkpoint" --pairs "$BASE/runs/$PROFILE/round1-pairs.jsonl" --validation "$BASE/data/validation.jsonl" --accumulation "$OPSD_ACCUMULATION" --output "$BASE/runs/$PROFILE/round1-opsd" --retriever-url http://10.120.17.101:18001/retrieve > "$BASE/logs/round1-opsd.log" 2>&1
export E1_MODEL="$BASE/runs/$PROFILE/round1-opsd/merged"
test -f "$E1_MODEL/config.json"
phase FRESH_GRPO_ROUND_2
export E1_STAGE=round-2-grpo
"$PYTHON" "$CONTROL/run_checked.py"
checkpoint="$BASE/runs/$PROFILE/round-2-grpo/actor/global_step_$steps"
test -f "$checkpoint/config.json"
"$PYTHON" "$CONTROL/verify_checkpoint.py" --initial "$E1_MODEL" --checkpoint "$checkpoint" --report "$BASE/receipts/$PROFILE-round2-checkpoint.json"
checkpoint=$(select_grpo 2 "$E1_MODEL")
phase OPSD_ROUND_2_TRAJECTORIES
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_2_trajectories-admission.json"
"$PYTHON" -m search_e1.rollout --model "$checkpoint" --data "$BASE/data/train.jsonl" --output "$BASE/runs/$PROFILE/round2-rollouts.jsonl" --k 5 --backend vllm --workers "$GPUS" --retriever-url http://10.120.17.101:18001/retrieve > "$BASE/logs/round2-rollouts.log" 2>&1
"$PYTHON" -m search_e1.mine --rollouts "$BASE/runs/$PROFILE/round2-rollouts.jsonl" --output "$BASE/runs/$PROFILE/round2-pairs.jsonl"
phase OPSD_ROUND_2
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/opsd_round_2-admission.json"
PYTHONPATH="$OPSD_RUNTIME/upstream/Search-R1:$OPSD_RUNTIME" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" -m search_e1.opsd --model "$checkpoint" --pairs "$BASE/runs/$PROFILE/round2-pairs.jsonl" --validation "$BASE/data/validation.jsonl" --accumulation "$OPSD_ACCUMULATION" --output "$BASE/runs/$PROFILE/round2-opsd" --retriever-url http://10.120.17.101:18001/retrieve > "$BASE/logs/round2-opsd.log" 2>&1
phase FINAL_SEVEN_BENCHMARK_EVALUATION
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/final_seven_benchmark_evaluation-admission.json"
"$PYTHON" -m search_e1.rollout --model "$BASE/runs/$PROFILE/round2-opsd/merged" --data "$BASE/data/test.jsonl" --output "$BASE/runs/$PROFILE/final-test.jsonl" --k 1 --backend vllm --workers "$GPUS" --temperature 0 --retriever-url http://10.120.17.101:18001/retrieve > "$BASE/logs/final-test.log" 2>&1
phase COMPLETED_ALL_STAGES

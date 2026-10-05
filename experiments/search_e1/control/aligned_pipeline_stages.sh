# The heldout selection may choose the initial model; validate the final trained
# checkpoint separately so weight-change evidence is not confused with selection.
select_grpo() {
 local round=$1 initial=$2 stage="$BASE/runs/$PROFILE/round-$1-grpo"
 "$PYTHON" "$CONTROL/select_grpo_checkpoint.py" --log "$stage/train.log" --validation "$BASE/data/validation.jsonl" --initial "$initial" --stage "$stage" --steps "$steps" --report "$BASE/receipts/$PROFILE-round$round-selection.json"
}
if [[ "$PROFILE" == r1 && -f "$BASE/receipts/continue-current-eight.json" ]]; then
 phase CONTINUING_GRPO_ROUND_1_FROM_STEP_3
else
 phase FRESH_GRPO_ROUND_1
fi
export E1_STAGE=round-1-grpo
"$PYTHON" "$CONTROL/run_checked.py"
steps=$E1_ALIGNED_STEPS
checkpoint="$BASE/runs/$PROFILE/round-1-grpo/actor/global_step_$steps"
test -f "$checkpoint/config.json"
"$PYTHON" "$CONTROL/verify_checkpoint.py" --initial "$BASE/assets/model" --checkpoint "$checkpoint" --report "$BASE/receipts/$PROFILE-round1-checkpoint.json"
checkpoint=$(select_grpo 1 "$BASE/assets/model")
phase INDEPENDENT_ROUND_1_EVALUATION
"$PYTHON" "$CONTROL/gpu_admission.py" > "$BASE/receipts/independent_round_1_evaluation-admission.json"
eval_steps=("$steps")
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

#!/usr/bin/env bash
# One complete algorithm stage per allocation; never interrupt a GRPO optimizer run.
set -euo pipefail
: "${PHASE:?Set PHASE to grpo1, opsd1, grpo2, opsd2 or eval}"
: "${RUN:?Set RUN to the shared experiment directory}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# Pending jobs retain their submitted sbatch script, including its old GPUS=4.
# Use the actual allocation when their GPU request is enlarged before starting.
export GPUS=${SLURM_GPUS_ON_NODE:-${GPUS:-4}}
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
export PYTHON PYTHONPATH="$ROOT/upstream/Search-R1:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
checkpoint() {
  local path="$RUN/round-$1/grpo/actor/global_step_200"
  test -f "$RUN/round-$1/grpo.done"
  test -f "$path/config.json"
  printf '%s\n' "$path"
}
case "$PHASE" in
  grpo1|grpo2)
    round=${PHASE#grpo}
    DIR="$RUN/round-$round"
    mkdir -p "$DIR"
    if [[ "$round" == 1 ]]; then
      MODEL="$ROOT/models/Qwen2.5-3B-Instruct"
    else
      test -f "$RUN/round-1/opsd.done"
      MODEL="$RUN/round-1/opsd/merged"
    fi
    test -f "$MODEL/config.json"
    test ! -e "$DIR/grpo.done"
    export GRPO_MICRO_PER_GPU=${GRPO_MICRO_PER_GPU:-2}
    export GRPO_SCORE_PER_GPU=${GRPO_SCORE_PER_GPU:-4}
    MODEL="$MODEL" OUTPUT="$DIR/grpo" STEPS=200 bash scripts/grpo.sh 2>&1 | tee "$DIR/grpo.log"
    test -f "$DIR/grpo/actor/global_step_200/config.json"
    touch "$DIR/grpo.done"
    ;;
  opsd1|opsd2)
    # Keep the global trajectory batch at 32 when the allocation grows.
    gpus=${GPUS:-4}
    if (( gpus < 1 || gpus > 32 || 32 % gpus != 0 )); then
      echo "OPSD GPU count must divide the global batch of 32: $gpus" >&2
      exit 2
    fi
    round=${PHASE#opsd}
    DIR="$RUN/round-$round"
    CHECKPOINT=$(checkpoint "$round")
    if [[ ! -f "$DIR/rollouts.done" ]]; then
      "$PYTHON" -m search_e1.rollout --model "$CHECKPOINT" --data data/train.jsonl --output "$DIR/rollouts.jsonl" --k 5 --backend vllm --workers "${GPUS:-4}" --retriever-url "$RETRIEVER_URL"
      touch "$DIR/rollouts.done"
    fi
    "$PYTHON" -m search_e1.mine --rollouts "$DIR/rollouts.jsonl" --output "$DIR/pairs.jsonl"
    test ! -e "$DIR/opsd.done"
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-4}" -m search_e1.opsd \
      --model "$CHECKPOINT" --pairs "$DIR/pairs.jsonl" --validation data/validation.jsonl \
      --accumulation "$((32 / gpus))" \
      --output "$DIR/opsd" --retriever-url "$RETRIEVER_URL" 2>&1 | tee "$DIR/opsd.log"
    test -f "$DIR/opsd/merged/config.json"
    touch "$DIR/opsd.done"
    ;;
  eval)
    test -f "$RUN/round-2/opsd.done"
    "$PYTHON" -m search_e1.rollout --model "$RUN/round-2/opsd/merged" --data data/test.jsonl \
      --output "$RUN/test.jsonl" --k 1 --backend vllm --workers "${GPUS:-4}" --temperature 0 --retriever-url "$RETRIEVER_URL"
    test -f "$RUN/test.jsonl.metrics.json"
    touch "$RUN/eval.done"
    ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac

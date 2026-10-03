#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?};PROFILE=${E1_PROFILE:?archive or r1};STAGE=${E1_STAGE:-smoke}
RUNTIME="$BASE/runtime/$PROFILE"
export PYTHON="$BASE/env/bin/python"
export PYTHONPATH="$RUNTIME/upstream/Search-R1:$RUNTIME"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export GPUS=4
export MODEL=${E1_MODEL:-$BASE/assets/model}
export OUTPUT="$BASE/runs/$PROFILE/$STAGE"
export DATA_DIR="$BASE/data"
export RETRIEVER_URL=${E1_RETRIEVER_URL:-http://10.123.4.11:18000/retrieve}
export GRPO_SCORE_PER_GPU=4
if [[ "$PROFILE" == archive ]]; then
 export STEPS=200 MINIBATCH=128 GRPO_MICRO_PER_GPU=2
 entropy=0.0
elif [[ "$PROFILE" == r1 ]]; then
 export STEPS=500 MINIBATCH=256 GRPO_MICRO_PER_GPU=16 GRPO_SCORE_PER_GPU=32
 entropy=0.001
else
 exit 2
fi
# Only A activates the archive's per-trajectory reduction. B preserves micro64.
if [[ "$PROFILE" == r1 ]]; then
 export E1_DISABLE_TRAJECTORY_BATCHING=1
fi
mkdir -p "$OUTPUT"
util=${E1_VLLM_UTIL:-0.4}
options=(actor_rollout_ref.actor.entropy_coeff="$entropy" actor_rollout_ref.rollout.gpu_memory_utilization="$util" ++trainer.val_before_train=true trainer.save_freq=50 trainer.test_freq=50)
if [[ "$STAGE" == smoke* ]]; then
 options+=(+trainer.stop_after=2 trainer.save_freq=1 ++trainer.val_before_train=false)
fi
if [[ "${E1_CONFIG_ONLY:-0}" == 1 ]]; then options+=(--cfg job); fi
cd "$RUNTIME"
bash scripts/grpo.sh "${options[@]}" > "$OUTPUT/train.log" 2>&1

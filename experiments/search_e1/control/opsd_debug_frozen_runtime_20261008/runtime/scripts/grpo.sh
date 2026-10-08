#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL:?Set MODEL to initial or merged Hugging Face checkpoint}"
: "${OUTPUT:?Set OUTPUT to round checkpoint directory}"
PYTHON=${PYTHON:-python}
GPUS=${GPUS:-8}
STEPS=${STEPS:-200}
BATCH=${BATCH:-512}
MINIBATCH=${MINIBATCH:-128}
DATA_DIR=${DATA_DIR:-$ROOT/data}
MICRO_PER_GPU=${GRPO_MICRO_PER_GPU:-1}
SCORE_PER_GPU=${GRPO_SCORE_PER_GPU:-1}
if (( MICRO_PER_GPU > 1 )) && [[ "${E1_DISABLE_TRAJECTORY_BATCHING:-0}" != 1 ]]; then
  export SEARCH_E1_TRAJECTORY_BATCHING=1
else
  export SEARCH_E1_TRAJECTORY_BATCHING=0
fi
export PYTHONPATH="$ROOT/upstream/Search-R1:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"$PYTHON" -m search_e1.grpo_main \
 data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/validation.parquet" \
 data.train_batch_size="$BATCH" data.val_batch_size=32 \
 data.max_prompt_length=4096 data.max_response_length=500 data.max_start_length=2048 data.max_obs_length=500 \
 algorithm.adv_estimator=grpo algorithm.no_think_rl=false \
 actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.external_lib=search_e1.worker_debug actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=true \
 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
 actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.state_masking=true \
 actor_rollout_ref.actor.ppo_mini_batch_size="$MINIBATCH" actor_rollout_ref.actor.ppo_micro_batch_size="$((GPUS * MICRO_PER_GPU))" \
 actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.grad_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.load_format=dummy_dtensor actor_rollout_ref.rollout.n_agent=5 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size="$((GPUS * SCORE_PER_GPU))" actor_rollout_ref.ref.log_prob_micro_batch_size="$((GPUS * SCORE_PER_GPU))" actor_rollout_ref.ref.fsdp_config.param_offload=true \
 trainer.logger=[console] trainer.n_gpus_per_node="$GPUS" trainer.nnodes=1 trainer.total_epochs=20 trainer.total_training_steps="$((STEPS + 1))" \
 trainer.save_freq=50 trainer.test_freq=50 trainer.default_hdfs_dir=null trainer.default_local_dir="$OUTPUT" \
 trainer.project_name=Search-E1 trainer.experiment_name=search-e1 \
 +trainer.val_before_train=false +trainer.val_only=false \
 max_turns=4 retriever.topk=3 retriever.url="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" "$@"

#!/usr/bin/env bash
set -euo pipefail
BASE=/data/search-g1
export SEARCH_G1_BASE="$BASE"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
unset PYTHONHOME RAY_ADDRESS
export C_INCLUDE_PATH=/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS PYTHONUNBUFFERED=1
export PROBE_URL=http://127.0.0.1:18012/judge_probe
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=probe ECO_PROBE_GATE=1
export ECO_BAL_GROUND=0.6 ECO_BAL_NOSEARCH_BONUS=0.5 ECO_W_COST=0.02
export ECO_WRONG_GROUND_SCALE=0 ECO_PROBE_BASE=0 ECO_PROBE_ALPHA_COST=0 ECO_USE_CF=0
export ECO_NO_SELF=1 ECO_PROBE_MAX_WAIT=300
cd "$BASE/repo"
python -m reproduction.a800.probe_admission --verify
python -m verl.trainer.main_ppo_eco \
 data.train_files="$BASE/data/train.parquet" data.val_files="$BASE/data/validation-64.parquet" \
 data.train_data_num=null data.val_data_num=null data.train_batch_size=128 data.val_batch_size=64 \
 data.max_prompt_length=4096 data.max_response_length=500 data.max_start_length=2048 data.max_obs_length=500 \
 data.shuffle_train_dataloader=True algorithm.adv_estimator=grpo \
 actor_rollout_ref.model.path="$BASE/assets/qwen2.5-3b-instruct" \
 actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=False \
 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
 actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.ppo_micro_batch_size=8 actor_rollout_ref.actor.use_dynamic_bsz=false \
 actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.grad_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
 actor_rollout_ref.rollout.log_prob_micro_batch_size=16 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.n_agent=5 actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.rollout.top_p=0.95 \
 actor_rollout_ref.ref.log_prob_micro_batch_size=16 actor_rollout_ref.ref.fsdp_config.param_offload=true \
 actor_rollout_ref.actor.state_masking=true algorithm.no_think_rl=false \
 "trainer.logger=['console']" +trainer.val_only=false +trainer.val_before_train=true \
 trainer.default_hdfs_dir=null trainer.n_gpus_per_node=4 trainer.nnodes=1 \
 trainer.save_freq=50 +trainer.acceptance_step=5 trainer.test_freq=50 \
 trainer.project_name=Search-G1 trainer.experiment_name=search-g1-a800 \
 trainer.total_epochs=15 trainer.total_training_steps=301 trainer.default_local_dir="$BASE/checkpoints/search-g1-a800" \
 max_turns=4 retriever.url=http://127.0.0.1:18010/retrieve retriever.topk=3 \
 2>&1 | tee -a "$SEARCH_G1_TRAIN_LOG"

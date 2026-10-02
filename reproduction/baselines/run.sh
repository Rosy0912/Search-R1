#!/usr/bin/env bash
set -euo pipefail
# This script does not choose GPUs or kill other jobs. Allocate an isolated job first.
METHOD=${1:?Usage: run.sh gigpo|tree_grpo|bapo|opsd|topd|cosearch [Hydra overrides]}
shift
: "${CUDA_VISIBLE_DEVICES:?Allocate an isolated GPU job and set CUDA_VISIBLE_DEVICES}"
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
overrides=("baselines.method=$METHOD")
case "$METHOD" in
    gigpo) ;;
    tree_grpo) overrides+=(actor_rollout_ref.rollout.n_agent=2) ;;
    bapo) overrides+=("baselines.bapo.stage_validation_files=${BAPO_STAGE_VALIDATION:?Set independent BAPO_STAGE_VALIDATION parquet}") ;;
    opsd)
        overrides+=("baselines.distillation.teacher_path=${OPSD_TEACHER:?Set frozen OPSD_TEACHER path}"
            baselines.distillation.privileged_answers=true
            actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0
            algorithm.kl_ctrl.kl_coef=0 "actor_rollout_ref.actor.ppo_micro_batch_size=${OPSD_GLOBAL_MICROBATCH:-8}") ;;
    topd)
        overrides+=("baselines.distillation.teacher_path=${TOPD_TEACHER:?Set distinct trained TOPD_TEACHER path}"
            baselines.distillation.privileged_answers=false
            actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0
            algorithm.kl_ctrl.kl_coef=0 actor_rollout_ref.actor.ppo_epochs=2) ;;
    cosearch) overrides+=("baselines.ranker.model_path=${COSEARCH_RANKER:?Set auxiliary COSEARCH_RANKER path}") ;;
    *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac
exec python -m search_r1.baselines.main "${overrides[@]}" "$@"

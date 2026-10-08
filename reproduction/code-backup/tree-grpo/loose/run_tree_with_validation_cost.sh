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
# Match the local Ray resource pool to the explicit allocation.
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
gpu_count=${#visible_gpus[@]}
run_root=${SEARCH_R1_BASELINE_RUN_ROOT:-/tmp/search-r1-baseline-${USER:-user}/$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$run_root"/{tmp,ray,cache}
export TMPDIR="$run_root/tmp" RAY_TMPDIR="$run_root/ray" XDG_CACHE_HOME="$run_root/cache"
# HF assets are explicit local paths, so isolated cache never triggers an implicit download.
export HF_HOME="$run_root/cache/hf" WANDB_MODE=offline
export SEARCH_R1_BASELINE_OUTPUT=${SEARCH_R1_BASELINE_OUTPUT:-$run_root/output}
overrides=("baselines.method=$METHOD" "trainer.n_gpus_per_node=$gpu_count")
case "$METHOD" in
    grpo|gigpo) ;;
    tree_grpo) overrides+=(actor_rollout_ref.rollout.n_agent=2) ;;
    bapo) ;;
    opsd)
        overrides+=("baselines.distillation.teacher_path=${OPSD_TEACHER:?Set frozen OPSD_TEACHER path}"
            baselines.distillation.privileged_answers=true
            actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0
            algorithm.kl_ctrl.kl_coef=0 "actor_rollout_ref.actor.ppo_micro_batch_size=${OPSD_GLOBAL_MICROBATCH:-$gpu_count}") ;;
    topd)
        overrides+=("baselines.distillation.teacher_path=${TOPD_TEACHER:?Set distinct trained TOPD_TEACHER path}"
            baselines.distillation.privileged_answers=false
            actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0
            algorithm.kl_ctrl.kl_coef=0 actor_rollout_ref.actor.ppo_epochs=2) ;;
    cosearch) overrides+=("baselines.ranker.model_path=${COSEARCH_RANKER:?Set auxiliary COSEARCH_RANKER path}") ;;
    *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac
# Hydra inspection does not touch GPU hardware or load production assets.
for arg in "$@"; do
    if [[ "$arg" == --cfg || "$arg" == --help || "$arg" == --info ]]; then
        exec python /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/validation_cost_entry.py "${overrides[@]}" "$@"
    fi
done
exec python -m search_r1.baselines.preflight -- python /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/validation_cost_entry.py "${overrides[@]}" "$@"

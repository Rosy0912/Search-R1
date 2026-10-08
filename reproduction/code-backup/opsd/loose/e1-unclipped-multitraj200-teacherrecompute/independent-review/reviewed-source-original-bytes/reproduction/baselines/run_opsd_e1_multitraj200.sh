#!/usr/bin/env bash
set -euo pipefail
# Fixed fresh multitraj200 profile; no arbitrary Hydra overrides or GPU selection.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
bootstrap="$repo_root/reproduction/baselines/e1_import_bootstrap.py"
env_repo=$(cd "${SEARCH_R1_REPO:-$repo_root}" && pwd -P)
[[ "$env_repo" == "$repo_root" ]] || { echo "SEARCH_R1_REPO differs from actual E1 source" >&2; exit 1; }
export SEARCH_R1_REPO="$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
mode=${1:---cfg}
[[ $# -le 1 ]] || { echo 'Only --cfg, --validate or --run is supported' >&2; exit 2; }
case "$mode" in
 --cfg) exec python "$bootstrap" --profile multitraj200 --mode cfg ;;
 --validate) exec python "$bootstrap" --profile multitraj200 --mode validate ;;
 --run) ;;
 *) echo 'Only --cfg, --validate or --run is supported' >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Fresh owned Slurm allocation required}"
: "${CUDA_VISIBLE_DEVICES:?Existing explicit allocation required}"
: "${SEARCH_R1_BASELINE_RUN_ROOT:?Explicit own absolute fresh run root required}"
[[ "$(hostname -s)" == h100-10 ]] || { echo 'E1 host differs' >&2; exit 1; }
IFS=',' read -r -a visible <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#visible[@]} == 8 ]] || { echo 'Exactly eight owned GPUs required' >&2; exit 1; }
[[ -d "$SEARCH_R1_BASELINE_RUN_ROOT" && ! -e "$SEARCH_R1_BASELINE_RUN_ROOT/resolved-profile.json" && ! -e "$SEARCH_R1_BASELINE_RUN_ROOT/train.log" ]] || { echo 'Owned fresh run directory required' >&2; exit 1; }
python "$bootstrap" --profile multitraj200 --mode validate > "$SEARCH_R1_BASELINE_RUN_ROOT/resolved-profile.json"
export SEARCH_R1_OPSD_LOCAL_HOST=h100-10
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
# The preflight checks real node/job/UUID/cgroup/resources and retains leases.
args=()
if [[ ${SEARCH_R1_GPU_PREFLIGHT:-0} == 1 ]]; then args+=(--reuse-inherited); fi
python "$bootstrap" --profile multitraj200 --mode preflight -- "${args[@]}" -- python "$bootstrap" --profile multitraj200 --mode run 2>&1 | tee "$SEARCH_R1_BASELINE_RUN_ROOT/train.log"

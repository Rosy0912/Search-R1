#!/usr/bin/env bash
# Submit externally, after review: one node h100-10, --gres=gpu:h100:8
# --exclusive --cpus-per-task=192 --mem=1T --time=7-00:00:00; srun --ntasks=1.
set -euo pipefail
for name in SLURM_JOB_ID SEARCH_R1_REPO E1_FORMAL_TOOLS E1_FORMAL_MANIFEST_SHA256 E1_NATIVE_IMPORT_CLOSURE E1_NATIVE_IMPORT_CLOSURE_SHA256 OPSD_E1_COMMON_DATA_ROOT OPSD_E1_COMMON_MANIFEST E1_TOOL_ASSET_MANIFEST OPSD_E1_REVIEW_PUBLIC_KEY OPSD_E1_REVIEW_PUBLIC_KEY_SHA256; do
 [[ -n "${!name:-}" ]] || { printf 'Missing externally reviewed input: %s\n' "$name" >&2; exit 2; }
done
export E1_ENTRY_SCRIPT="$0"
source /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/activate
export PYTHONPATH="$SEARCH_R1_REPO" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=4 VLLM_ATTENTION_BACKEND=XFORMERS
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF PYTHONOPTIMIZE
export E1_FORMAL_RUN=/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-$SLURM_JOB_ID
export SEARCH_R1_BASELINE_RUN_ROOT="$E1_FORMAL_RUN"
[[ ! -e "$E1_FORMAL_RUN" && ! -L "$E1_FORMAL_RUN" ]]
mkdir "$E1_FORMAL_RUN"
trap 'rc=$?; printf "%s\n" "$rc" > "$E1_FORMAL_RUN/job-exit.txt"' EXIT
exec 9>/mnt/data/shaoshitong/search-r1-rl/opsd/gpu-job.lock
flock -n 9
cd "$SEARCH_R1_REPO"
# Bash remains alive with FD9 while a same-PID supervisor exec chain acquires GPU leases.
python "$E1_FORMAL_TOOLS/formal_supervisor.py" --prepare 2>&1 | tee "$E1_FORMAL_RUN/formal-wrapper.log"

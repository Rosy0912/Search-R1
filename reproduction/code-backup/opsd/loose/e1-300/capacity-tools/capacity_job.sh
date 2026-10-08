#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Owned fresh job required}"
: "${SEARCH_R1_REPO:?Frozen new repository required}"
: "${E1_CAPACITY_TOOLS:?Externally reviewed tools directory required}"
: "${E1_CAPACITY_MANIFEST_SHA256:?Reviewed manifest pin required}"
: "${E1_TOOL_ASSET_MANIFEST:?Full retrieval asset manifest required}"
: "${E1_TOOL_ASSET_MANIFEST_SHA256:?Externally reviewed full assets pin required}"
: "${E1_EXPECTED_SOURCE:?Externally reviewed exact new source required}"
source /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/activate
export PYTHONPATH="$SEARCH_R1_REPO" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=4
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export E1_CAPACITY_RUN=/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/capacity-$SLURM_JOB_ID
[[ ! -e "$E1_CAPACITY_RUN" && ! -L "$E1_CAPACITY_RUN" ]]
mkdir "$E1_CAPACITY_RUN"
trap 'rc=$?; printf "%s\n" "$rc" > "$E1_CAPACITY_RUN/job-exit.txt"' EXIT
exec 9>/mnt/data/shaoshitong/search-r1-rl/opsd/gpu-job.lock
flock -n 9
cd "$SEARCH_R1_REPO"
python "$E1_CAPACITY_TOOLS/capacity_entry.py" --prepare
python -m search_r1.baselines.preflight -- python "$E1_CAPACITY_TOOLS/capacity_entry.py" --run 2>&1 | tee "$E1_CAPACITY_RUN/capacity.log"

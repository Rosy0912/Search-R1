#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Own allocation required}"
root=/mnt/data/shaoshitong/search-r1-rl/opsd
run="$root/runs/opsd-numeric-diagnostic-$SLURM_JOB_ID"
[[ ! -e "$run" && ! -L "$run" ]]
mkdir "$run"
trap 'rc=$?; printf "%s\n" "$rc" > "$run/exit.txt"' EXIT
export OPSD_EVIDENCE="$run" OPSD_TEACHER="$root/assets-root/assets/qwen2.5-3b"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source "$root/env/bin/activate"
cd "$root/repo"
[[ $(git rev-parse HEAD) == 3e3fb94d465b16dd0ecad5e554eb6f12a4991883 ]]
git rev-parse HEAD > "$run/source.txt"
sha256sum "$root/tools/runtime/gpu_teacher_probe-diagnostic.py" "$0" > "$run/scripts.sha256"
scontrol show job "$SLURM_JOB_ID" > "$run/slurm-job.txt"
python -u "$root/tools/runtime/resource_sample.py" "$run/resources.ndjson" $$ &
python -m search_r1.baselines.preflight -- python -u "$root/tools/runtime/gpu_teacher_probe-diagnostic.py" --output "$run/probe.json" --fp32-reference --flash-deterministic --factor-bf16-backward 2>&1 | tee "$run/probe.log"

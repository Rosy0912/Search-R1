#!/bin/bash
set -euo pipefail
ROOT=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009
SOURCE=080ebbc46b89d61567cd33e1a5e2f1b40afd8b84
TOKEN=${1:?unique hardware token required}
[[ "$TOKEN" =~ ^[a-f0-9]{32}$ && $(hostname) = h100-5 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
[[ ${SLURM_RESTART_COUNT:-0} = 0 && -n ${CUDA_VISIBLE_DEVICES:-} ]]
export SPT_NOENV=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$ROOT/env/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo"
unset PYTHONOPTIMIZE LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$ROOT/cache/hf" HF_DATASETS_CACHE="$ROOT/cache/datasets" XDG_CACHE_HOME="$ROOT/cache"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export TRITON_LIBCUDA_PATH="$ROOT/lib/driver" LIBRARY_PATH="$ROOT/lib/driver"
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
"$ROOT/env/bin/python" -B - <<'PYLOCK'
from pathlib import Path
import os, stat
root=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
for name in ('runs','logs','tools','tools/deployment','repo','env'):
 path=root/name
 if not path.is_dir() or path.resolve()!=path:raise RuntimeError('Canonical own worker directories required')
head=root/'scheduler-head';lock=head/'execution-global.lock'
if root.resolve()!=root or head.resolve()!=head or lock.resolve()!=lock or lock.is_symlink():
 raise RuntimeError('Canonical own worker lock required')
if lock.exists():
 st=lock.stat()
 if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_uid!=os.getuid():raise RuntimeError('Own ordinary single-link lock required')
PYLOCK
exec 9>>"$ROOT/scheduler-head/execution-global.lock"
flock -n 9 || { echo 'Another new-source worker owns execution' >&2; exit 3; }
[[ $(git -C "$ROOT/repo" rev-parse HEAD) = "$SOURCE" && -z $(git -C "$ROOT/repo" status --porcelain) ]]
RUN="$ROOT/runs/hardware-aligned-em200-$TOKEN-$SLURM_JOB_ID"
mkdir "$RUN"
"$ROOT/env/bin/python" -B - "$RUN" <<'PYRUN'
from pathlib import Path
import sys
run=Path(sys.argv[1])
if not run.is_dir() or run.resolve()!=run:raise RuntimeError('Canonical new own RUN required')
PYRUN
export BAPO_RUN_ID="$RUN" SEARCH_R1_REPO="$ROOT/repo" SEARCH_R1_ASSETS="$ROOT/assets-root"
printf '%s\n' "$SOURCE" > "$RUN/source-commit.txt"
cleanup() {
 status=$?; trap - EXIT
 cleanup_status=0
 "$ROOT/env/bin/python" -B "$ROOT/tools/owned_processes.py" stop --run "$RUN" > "$RUN/own-process-cleanup.json" 2> "$RUN/own-process-cleanup.err" || cleanup_status=$?
 nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$RUN/gpu-exit.csv" 2> "$RUN/gpu-exit.err" || cleanup_status=1
 printf '%s\n' "$status" > "$RUN/original-exit-code.txt"
 printf '%s\n' "$cleanup_status" > "$RUN/cleanup-exit-code.txt"
 [[ "$status" != 0 || "$cleanup_status" = 0 ]] || status=$cleanup_status
 exit "$status"
}
trap cleanup EXIT
cd "$ROOT/repo"
"$ROOT/env/bin/python" -B -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/resource_watch.py" --run "$RUN" -- bash "$ROOT/tools/deployment/hardware_inner.sh" > "$RUN/hardware.log" 2>&1

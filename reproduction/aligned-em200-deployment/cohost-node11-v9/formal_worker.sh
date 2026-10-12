#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v9-20261009
OLDROOT=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009
SOURCE=4090e4e8f961ce4ef2057e85dd5a3e6a9506094a
TOKEN=${1:?unique token}
[[ "$TOKEN" =~ ^[a-f0-9]{32}$ && $(hostname) = h100-11 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_RESTART_COUNT:-0} = 0 && -n ${CUDA_VISIBLE_DEVICES:-} ]]
export SPT_NOENV=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$ROOT/env/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$ROOT/tools:$ROOT/retriever-gpu-lib:$ROOT/repo"
unset PYTHONOPTIMIZE LD_LIBRARY_PATH http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$ROOT/cache/hf" HF_DATASETS_CACHE="$ROOT/cache/datasets" XDG_CACHE_HOME="$ROOT/cache"
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
export TRITON_LIBCUDA_PATH="$ROOT/lib/driver" LIBRARY_PATH="$ROOT/lib/driver"
"$ROOT/env/bin/python" -B - "$OLDROOT/scheduler-head/execution-global.lock" <<'PY'
from pathlib import Path
import os,stat,sys
p=Path(sys.argv[1]);assert p.is_file() and p.parent.resolve()==p.parent and p.resolve()==p and not p.is_symlink()
if p.exists():
 s=p.stat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid()
PY
exec 9>>"$OLDROOT/scheduler-head/execution-global.lock"
flock -n 9 || { echo 'Another formal worker owns shared execution lock' >&2; exit 3; }
RUN="$ROOT/runs/formal-aligned-em200-$TOKEN-$SLURM_JOB_ID"
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_admission.py" verify
mkdir "$RUN"
export BAPO_RUN_ID="$RUN" SEARCH_R1_REPO="$ROOT/repo" SEARCH_R1_ASSETS="$ROOT/assets-root"
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/formal_admission.py" claim --run "$RUN"
cleanup() {
 status=$?; trap - EXIT
 cleanup_status=0
 "$ROOT/env/bin/python" -B "$ROOT/tools/resource_watch.py" --cleanup-run "$RUN" > "$RUN/cleanup-command.log" 2>&1 || cleanup_status=$?
 nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$RUN/gpu-exit.csv" 2> "$RUN/gpu-exit.err" || cleanup_status=1
 printf '%s\n' "$status" > "$RUN/original-exit-code.txt"
 printf '%s\n' "$cleanup_status" > "$RUN/cleanup-exit-code.txt"
 [[ "$status" != 0 || "$cleanup_status" = 0 ]] || status=$cleanup_status
 exit "$status"
}
trap cleanup EXIT
cd "$ROOT/repo"
"$ROOT/env/bin/python" -B "$ROOT/tools/deployment/cohost-node11-v9/bootstrap.py" --run "$RUN" --minimum-start-free-mib 30720 --reserve-free-mib 1024 -- bash "$ROOT/tools/deployment/cohost-node11-v9/formal_inner.sh" > "$RUN/formal-supervisor.log" 2>&1

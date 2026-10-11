#!/usr/bin/env bash
set -euo pipefail
export RECURSIVE_BRAN_BASE=/data/shaoshitong/recursive-bran-a-8gpu-20261011
cd "$RECURSIVE_BRAN_BASE/source"
python=/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export RAY_ADDRESS=local
export TMPDIR="$RECURSIVE_BRAN_BASE/tmp" RAY_TMPDIR="$RECURSIVE_BRAN_BASE/ray" XDG_CACHE_HOME="$RECURSIVE_BRAN_BASE/cache"
export HF_HOME="$RECURSIVE_BRAN_BASE/cache/hf"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"
# Reject any current compute PID except the explicitly retained own retriever.
"$python" - <<'PY'
import subprocess,pathlib
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
for line in rows.splitlines():
 pid=int(line.strip());assert pid==2239529,'Unexpected GPU process; no signal or takeover'
 p=pathlib.Path('/proc',str(pid));assert p.stat().st_uid==1005
 st=(p/'stat').read_text().split(') ',1)[1].split();assert int(st[19])==914664099
PY
exec "$python" -m search_r1.baselines.preflight -- "$python" release/recursive_bran_a/train.py

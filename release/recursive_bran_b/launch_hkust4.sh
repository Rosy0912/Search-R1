#!/usr/bin/env bash
set -euo pipefail
export RECURSIVE_BRAN_BASE=/weka_3/shaoshitong/recursive-bran-b-4gpu-20261011
cd "${RECURSIVE_BRAN_SOURCE:?explicit immutable source required}"
python=/weka_3/shaoshitong/search-e1-config-study-20261004/env/bin/python
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VLLM_ATTENTION_BACKEND=XFORMERS HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export RAY_ADDRESS=local
export RECURSIVE_BRAN_STRICT_GPU_GATE=1
export RECURSIVE_BRAN_EXPERIMENT_LOCK=/weka_3/shaoshitong/search-e1-config-study-20261004/experiment.lock
export TMPDIR="$RECURSIVE_BRAN_BASE/tmp" RAY_TMPDIR="$RECURSIVE_BRAN_BASE/ray" XDG_CACHE_HOME="$RECURSIVE_BRAN_BASE/cache"
export HF_HOME="$RECURSIVE_BRAN_BASE/cache/hf"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"
"$python" - <<'PY'
import subprocess,pathlib,json,urllib.request
selected={'GPU-ed24344f-9cd8-6d29-e8b9-d7582a8cfd65','GPU-79de0d51-c2a2-897e-f76a-ec84f03b38e2','GPU-3a89b647-73fb-27bc-c8a8-e9f663b96d5f','GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3'}
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
for line in rows.splitlines():
 gpu,pid,mem=[v.strip() for v in line.split(',')]
 if gpu not in selected:continue
 pid=int(pid);assert pid==1749337 and gpu=='GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3' and float(mem)<=1024,'Selected GPU is occupied; no takeover'
 p=pathlib.Path('/proc',str(pid));assert p.stat().st_uid==1001
 st=(p/'stat').read_text().split(') ',1)[1].split();assert int(st[19])==641133475
health=json.load(urllib.request.urlopen('http://10.123.4.11:18003/health',timeout=15))
assert health['ready'] is True,'Retriever not ready'
assert health['ntotal']==21015324 and health['d']==768 and health['tf32'] is False and health['query_micro']==64
assert health['source_manifest_sha256']=='b1fe4576cd1d3314fe35fde3443cf63105110cf6dc9a71daa9e807f2cd2c4508' and health['acceptance_sha256']=='f256ee31dea1297a07baa28ca9b722e678fbc3555669a3d2bcebcb00884a9443','Retriever numeric identity changed'
print('REMOTE_RETRIEVER_HEALTH',json.dumps(health),flush=True)
PY
exec "$python" -m search_r1.baselines.preflight -- "$python" release/recursive_bran_b/train.py

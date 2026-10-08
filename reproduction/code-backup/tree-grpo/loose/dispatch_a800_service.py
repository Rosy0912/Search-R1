import pathlib,json,subprocess,os,time,hashlib
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');tag='cpu-encoder-fp32-http-20261003'
assert (root/'logs/preparation-completed.json').exists()
marker=root/'logs'/f'{tag}-dispatched.json'
with marker.open('x') as f:json.dump({'epoch':time.time(),'platform_cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'scope':'candidate encoder+HTTP gate only, no training acceptance','source_sha256':hashlib.sha256((root/'tools/aux_retrieval_server.py').read_bytes()).hexdigest()},f)
runner=root/'tools'/f'{tag}-runner.sh'
runner.write_text('''#!/bin/bash
set -uo pipefail
ROOT=/data/shaoshitong/search-r1-rl/tree-grpo-retrieval
mkdir -p "$ROOT/cache/hf" "$ROOT/cache/datasets" "$ROOT/tmp"
export CUDA_VISIBLE_DEVICES=GPU-c63282c6-1d05-224e-234b-2a8c794b1f04
export PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HOME="$ROOT/cache/hf" HF_DATASETS_CACHE="$ROOT/cache/datasets" TMPDIR="$ROOT/tmp" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
"$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/aux_retrieval_server.py" --run cpu-encoder-fp32-http-20261003 --port 18082 --max-seconds 14400
rc=$?
echo "$rc" > "$ROOT/logs/cpu-encoder-fp32-http-20261003.exit"
exit "$rc"
''')
with (root/'logs'/f'{tag}.log').open('x') as out,open('/dev/null') as dev:
 child=subprocess.Popen(['bash',str(runner)],stdin=dev,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 (root/'logs'/f'{tag}.pid').write_text(str(child.pid)+'\n')
print(json.dumps({'runner_pid':child.pid,'tag':tag,'scope':'bounded candidate CPU encoder+single auxiliary GPU FP32 retrieval HTTP gate'}))

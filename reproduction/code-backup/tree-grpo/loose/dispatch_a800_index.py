import pathlib,json,subprocess,time,hashlib
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');tag='fp32-index-http-20261003'
marker=root/'logs'/f'{tag}-dispatched.json'
with marker.open('x') as f:json.dump({'epoch':time.time(),'scope':'binary index candidate only; no encoder or production training acceptance','source_sha256':hashlib.sha256((root/'tools/index_gpu_server.py').read_bytes()).hexdigest()},f)
runner=root/'tools'/f'{tag}-runner.sh'
runner.write_text('''#!/bin/bash
set -uo pipefail
ROOT=/data/shaoshitong/search-r1-rl/tree-grpo-retrieval
export CUDA_VISIBLE_DEVICES=GPU-c63282c6-1d05-224e-234b-2a8c794b1f04
export PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$ROOT"
"$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/index_gpu_server.py" --run fp32-index-http-20261003 --port 18082 --max-seconds 14400
rc=$?
echo "$rc" > "$ROOT/logs/fp32-index-http-20261003.exit"
exit "$rc"
''')
with (root/'logs'/f'{tag}.log').open('x') as out,open('/dev/null') as dev:
 child=subprocess.Popen(['bash',str(runner)],stdin=dev,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 (root/'logs'/f'{tag}.pid').write_text(str(child.pid)+'\n')
print(json.dumps({'runner_pid':child.pid,'tag':tag}))

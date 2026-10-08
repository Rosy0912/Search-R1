import pathlib,json,subprocess,os,time,hashlib
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');tag='fp32-index-20261003-0754'
assert (root/'logs/preparation-completed.json').exists()
marker=root/'logs'/f'{tag}-dispatched.json'
with marker.open('x') as f:json.dump({'epoch':time.time(),'platform_cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'hostname':subprocess.check_output(['hostname'],text=True).strip(),'nvidia_visible_devices':os.environ.get('NVIDIA_VISIBLE_DEVICES'),'scope':'auxiliary GPU4 only within existing platform allocation; PLAN permits unassigned retrieval GPU','probe_sha256':hashlib.sha256((root/'tools/a800_index_probe.py').read_bytes()).hexdigest()},f)
runner=root/'tools'/f'{tag}-runner.sh'
runner.write_text('''#!/bin/bash
set -uo pipefail
ROOT=/data/shaoshitong/search-r1-rl/tree-grpo-retrieval
export CUDA_VISIBLE_DEVICES=GPU-c63282c6-1d05-224e-234b-2a8c794b1f04
export PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$ROOT"
"$ROOT/env/bin/python" -m search_r1.baselines.preflight -- timeout --signal=TERM 1800 "$ROOT/env/bin/python" "$ROOT/tools/a800_index_probe.py" fp32-index-20261003-0754
rc=$?
echo "$rc" > "$ROOT/logs/fp32-index-20261003-0754.exit"
exit "$rc"
''')
with (root/'logs'/f'{tag}.log').open('x') as out,open('/dev/null') as dev:
 p=subprocess.Popen(['bash',str(runner)],stdin=dev,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 (root/'logs'/f'{tag}.pid').write_text(str(p.pid)+'\n')
print(json.dumps({'runner_pid':p.pid,'tag':tag,'scope':'one bounded single auxiliary GPU probe; no HTTP server or training'}))

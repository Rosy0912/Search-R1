import pathlib,json,subprocess,time,hashlib,argparse,re,math
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');tag='fp32-index-formal-20261003'
p=argparse.ArgumentParser();p.add_argument('--five-acceptance-sha256',required=True);a=p.parse_args()
assert re.fullmatch('[0-9a-f]{64}',a.five_acceptance_sha256)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'tools/index_gpu_server.py';assert digest(source)=='a50654c46e816bd7084992dfaef5cb70b802dd4499c0e9bb19cf04bc0391b5a5'
accepted=root/'inputs/five-stage-acceptance-reviewed.json';assert digest(accepted)==a.five_acceptance_sha256
five=json.loads(accepted.read_text());assert five['status']=='REAL_TREE_STAGE_ACCEPTED' and five['stage']=='five' and five['actual_outer_updates']==5
assert digest(root/'inputs/five-real-stage-independent-review.md')==five['evidence_file_sha256'][five['independent_review_file']]
for path in [root/'runs'/tag,root/'tools'/f'{tag}-runner.sh']+[root/'logs'/f'{tag}{suffix}' for suffix in ['-dispatched.json','.log','.pid','.exit']]:assert not path.exists() and not path.is_symlink(), 'Existing own formal tag artifact'
oldrun=root/'runs/fp32-index-http-20261003';oldcost=json.loads((oldrun/'cost.json').read_text());oldprov=json.loads((oldrun/'provenance.json').read_text())
assert oldprov['pid']==774407 and oldprov['run']==str(oldrun) and oldprov['source_sha256']==digest(source)
assert oldcost['allocated_auxiliary_gpus']==1 and oldcost['gpu_uuid']==oldprov['gpu_uuid']=='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
assert oldcost['start_epoch']<=oldcost['end_epoch']<=time.time() and oldcost['end_epoch']>oldcost['start_epoch']
for key in ['start_epoch','end_epoch','gpu_peak_used_bytes','self_cpu_seconds','reaped_child_cpu_seconds','self_peak_rss_bytes']:assert isinstance(oldcost[key],(int,float)) and math.isfinite(oldcost[key]) and oldcost[key]>=0
assert isinstance(oldcost['violations'],list) and isinstance(oldcost['scope'],str) and oldcost['scope']
assert (root/'logs/fp32-index-http-20261003.exit').read_text().strip().isdigit()
# Never replace the accepted smoke instance while training still uses it.
import os,socket
try: os.kill(774407,0)
except ProcessLookupError: pass
else: raise RuntimeError('Old accepted index process remains alive; no second service launch')
assert (root/'runs/fp32-index-http-20261003/cost.json').is_file()
with socket.socket() as check: check.bind(('127.0.0.1',18082))
marker=root/'logs'/f'{tag}-dispatched.json'
with marker.open('x') as f:json.dump({'epoch':time.time(),'scope':'new longer-lived FP32 index candidate; requires new binary/HTTP gate and five-stage acceptance before formal use','max_seconds':777600,'source_sha256':digest(source),'previous_auxiliary_cost_sha256':digest(oldrun/'cost.json'),'previous_auxiliary_provenance_sha256':digest(oldrun/'provenance.json'),'five_acceptance_sha256':a.five_acceptance_sha256,'five_independent_review_sha256':digest(root/'inputs/five-real-stage-independent-review.md')},f)
runner=root/'tools'/f'{tag}-runner.sh'
runner_text='''#!/bin/bash
set -uo pipefail
ROOT=/data/shaoshitong/search-r1-rl/tree-grpo-retrieval
export CUDA_VISIBLE_DEVICES=GPU-c63282c6-1d05-224e-234b-2a8c794b1f04
export PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$ROOT"
"$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/index_gpu_server.py" --run fp32-index-formal-20261003 --port 18082 --max-seconds 777600
rc=$?
echo "$rc" > "$ROOT/logs/fp32-index-formal-20261003.exit"
exit "$rc"
'''
with runner.open('x') as f:f.write(runner_text)
with (root/'logs'/f'{tag}.log').open('x') as out,open('/dev/null') as dev:
 child=subprocess.Popen(['bash',str(runner)],stdin=dev,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 with (root/'logs'/f'{tag}.pid').open('x') as f:f.write(str(child.pid)+'\n')
print(json.dumps({'runner_pid':child.pid,'tag':tag}))

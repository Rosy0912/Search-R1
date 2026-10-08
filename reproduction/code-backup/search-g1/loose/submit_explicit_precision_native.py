"""Submit one new immutable source-bound native smoke after strict review/tests."""
import sys,json,subprocess,shlex,uuid,base64,time,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
root=Path('/Users/shaoshitong/Documents/Search-G1');repo=root/'repo';prep=root/'preparation'
assert not (prep/'explicit-precision-native-launch.json').exists(),'Inspect the existing submitted handle before any further submission'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
# A separate coalescer may remain untracked, but deployed tracked sources must be committed.
assert not any(line and not line.startswith('??') for line in status.splitlines()),status
names=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
files={n:(repo/n).read_bytes() for n in names if n and Path(n).suffix in ('.py','.sh','.yaml','.yml','.json') and 'evidence' not in Path(n).parts}
hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
test=json.loads((prep/'explicit-precision-cpu-result.json').read_text());assert test['returncode']==0
for n,h in test['source_sha256'].items():assert hashes[n]==h,(n,'Changed after target CPU test')
base='/dataset/shaoshitong/Search-G1-common8-explicit-fp32-'+uuid.uuid4().hex[:12]
script='''#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?}"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export SEARCH_G1_PROFILE=common8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" XDG_CACHE_HOME="$BASE/cache" HF_HOME="$BASE/cache/hf"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
unset PYTHONHOME RAY_ADDRESS
cd "$BASE/repo"
python -m reproduction.common8.prepare_run --base "$BASE" --model "$BASE/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725" --official-manifest "$BASE/preparation/official-model-manifest-original.json" --source-commit @COMMIT@
python - <<'ADMIT'
import os
from reproduction.common8.native_allocation import admit
print(admit(os.environ['SEARCH_G1_BASE']))
ADMIT
exec python -m torch.distributed.run --standalone --nproc_per_node=8 -m reproduction.common8.model_smoke --phase initial
'''.replace('@COMMIT@',commit)
payload=dict(base=base,source_commit=commit,files={n:base64.b64encode(v).decode() for n,v in files.items()},sha256=hashes,script=script)
claim=prep/'explicit-precision-native-submit-intent.json';assert not claim.exists(),'Existing submit intent: inspect its namespace and claim; do not silently resubmit'
claim.write_text(json.dumps(dict(base=base,source_commit=commit,source_sha256=hashes,created_unix=time.time()),indent=2))
code=r'''
import os,json,subprocess,sys,hashlib,time;from pathlib import Path
payload=json.load(sys.stdin);b=Path(payload['base']);old=Path('/dataset/shaoshitong/Search-G1-common8-attempt2-39e7968c7831')
assert not b.exists(),'Observe the existing namespace before retrying'
b.mkdir()
for n in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/n).mkdir(parents=True,exist_ok=True)
for n in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):(b/n).symlink_to(old/n,target_is_directory=True)
for n,value in payload['files'].items():
 p=b/'repo'/n;assert p.is_relative_to(b/'repo');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(__import__('base64').b64decode(value));assert hashlib.sha256(p.read_bytes()).hexdigest()==payload['sha256'][n]
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):os.link(old/'data'/n,b/'data'/n)
(b/'preparation/official-model-manifest-original.json').write_bytes((old/'preparation/official-model-manifest-original.json').read_bytes())
(b/'preparation/job.sh').write_text(payload['script'])
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=01:00:00','--job-name=search-g1-explicit-fp32-smoke','--chdir='+str(b/'repo'),'--output='+str(b/'logs/slurm-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b),str(b/'preparation/job.sh')]
with (b/'preparation/submit-claim.json').open('x') as f:json.dump(dict(base=str(b),scope='source-bound native initial model/precision smoke only; no trainer',created_unix=time.time()),f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr
result=dict(base=str(b),job_id=r.stdout.strip().split(';')[0],source_commit=payload['source_commit'],source_sha256=payload['sha256'],actual_grpo_updates=0,scope='Explicit recipe/scoped compute precision/actual runtime dtype native model smoke; no global monkeypatch or trainer',submitted_unix=time.time())
(b/'preparation/launch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),input=json.dumps(payload),text=True,capture_output=True,timeout=90);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(prep/'explicit-precision-native-launch.json').write_text(json.dumps(x,indent=2));print(json.dumps({k:v for k,v in x.items() if k!='source_sha256'},indent=2))

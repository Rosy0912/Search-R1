import sys,json,subprocess,shlex,uuid,base64,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');repo=prep.parent/'repo'
prior=json.loads((prep/'safety-target-cpu-result-v2.json').read_text());assert prior['returncode']==0
expected=prior['sha256']
for f,h in expected.items():assert hashlib.sha256((repo/f).read_bytes()).hexdigest()==h,f
native='reproduction/common8/test_policy_safety_native.py';content=base64.b64encode((repo/native).read_bytes()).decode();expected[native]=hashlib.sha256((repo/native).read_bytes()).hexdigest()
base='/dataset/shaoshitong/Search-G1-common8-safety-component-'+uuid.uuid4().hex[:12]
script='''#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?}"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo" SEARCH_G1_PROFILE=common8 SEARCH_G1_COMPONENT_FIXTURE=1
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
export C_INCLUDE_PATH="/dataset/shaoshitong/Search-G1-common8/bootstrap-tools/python-dev/usr/include/python3.10:/dataset/shaoshitong/Search-G1-common8/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" TMPDIR="$BASE/tmp"
unset PYTHONHOME RAY_ADDRESS
cd "$BASE/repo"
python - <<'ADMIT'
import os,json
from pathlib import Path
from reproduction.common8.native_allocation import admit
b=Path(os.environ['SEARCH_G1_BASE'])
p=b/'logs/run-manifest.json';p.write_text(json.dumps(dict(profile='common8',run_id='component-bootstrap',completed_updates=0,role='native_component_bootstrap')))
print(admit(b));p.rename(b/'logs/native-component-bootstrap-manifest.json')
ADMIT
exec python -m torch.distributed.run --standalone --nproc_per_node=8 -m reproduction.common8.test_policy_safety_native
'''
code='import os,json,shutil,subprocess,time,hashlib,base64;from pathlib import Path\nb=Path('+repr(base)+');old=Path('+repr(prior['base'])+');expected='+repr(expected)+';content='+repr(content)+';script='+repr(base64.b64encode(script.encode()).decode())+'\n'+'''
assert json.loads((old/'safety-cpu-result-v2.json').read_text())['returncode']==0
b.mkdir(exist_ok=False)
for name in ('logs','preparation','tmp','cache/triton','cache/inductor'):(b/name).mkdir(parents=True)
shutil.copytree(old,b/'repo',ignore=shutil.ignore_patterns('__pycache__','.git'))
(b/'env').symlink_to('/dataset/shaoshitong/Search-G1-common8/env',target_is_directory=True)
(b/'repo/reproduction/common8/test_policy_safety_native.py').write_bytes(base64.b64decode(content))
for f,h in expected.items():assert hashlib.sha256((b/'repo'/f).read_bytes()).hexdigest()==h,f
(b/'preparation/source-sha256.json').write_text(json.dumps(expected,indent=2))
(b/'preparation/job.sh').write_bytes(base64.b64decode(script))
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=16','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=00:30:00','--job-name=search-g1-safety-component','--chdir='+str(b/'repo'),'--output='+str(b/'logs/slurm-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b),str(b/'preparation/job.sh')]
with (b/'preparation/submit-claim.json').open('x') as f:json.dump(dict(scope='synthetic FSDP actual actor-loop component only, not production calibration',unix=time.time()),f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr
x=dict(job_id=r.stdout.strip().split(';')[0],base=str(b),source_commit='276842f',source_sha256=expected,actual_grpo_updates=0,production_admission=False,argv=args,submitted_unix=time.time());(b/'preparation/launch.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(prep/'safety-native-launch.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))

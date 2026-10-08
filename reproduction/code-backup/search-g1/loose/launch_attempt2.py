import sys,json,shlex,subprocess,time,uuid,base64,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');repo=prep.parent/'repo';prepared=json.loads((prep/'attempt2-preparation.json').read_text());base=prepared['base'];commit=prepared['source_commit']
def remote(host,code,python='python3'):
 compile(code,'remote','exec');r=subprocess.run(control.route(host,python+' -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr;return json.loads(r.stdout)
code='import json,os,time;from pathlib import Path\nb=Path('+repr(base)+')\n'+r'''
old=Path('/dataset/shaoshitong/Search-G1-common8');ledger=json.loads((old/'logs/process-ledger.json').read_text())
for e in [e for job in ledger['jobs'].values() for e in job['ledger'].values()]:
 p=Path('/proc')/str(e['pid'])
 if p.exists():
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  assert int(s[19])!=e['start_ticks'] or s[0]=='Z', 'old owned child still alive'
import accelerate,matplotlib,pybind11,torch
from accelerate import init_empty_weights
with init_empty_weights():m=torch.nn.Linear(4,4)
assert m.weight.device.type=='meta'
assert not (b/'logs/run-manifest.json').exists()
print(json.dumps(dict(accepted=True,checked_unix=time.time(),scope='target CPU imports/meta context and old owned process retirement; not GPU model admission')))
'''
p=remote('h100-8',code,base+'/env/bin/python');(prep/'attempt2-prelaunch-cpu.json').write_text(json.dumps(p,indent=2))
export='/dataset/shaoshitong/g1-common8-original-retrieval-export-attempt2-'+uuid.uuid4().hex[:12]
source=(repo/'reproduction/common8/assets_export.py').read_text().replace("BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')",'BASE=Path('+repr(export)+')')
code='import base64,json,os,subprocess,time,hashlib;from pathlib import Path\nb=Path('+repr(export)+');b.mkdir(exist_ok=False);s=b/"serve_original_retrieval.py";s.write_bytes(base64.b64decode('+repr(base64.b64encode(source.encode()).decode())+'))\n'+r'''
env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (b/'server.log').open('x') as f:p=subprocess.Popen(['nice','-n','19','python3',str(s)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 if (b/'handle.json').exists():break
 assert p.poll() is None;time.sleep(.1)
h=json.loads((b/'handle.json').read_text());assert h['pid']==p.pid
h.update(export=str(b),source_sha256=hashlib.sha256(s.read_bytes()).hexdigest());print(json.dumps(h))
'''
h=remote('h100-9',code);(prep/'attempt2-retrieval-export-handle.json').write_text(json.dumps(h,indent=2))
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=3-00:00:00','--job-name=search-g1-common8-attempt2','--chdir='+base+'/repo','--output='+base+'/logs/native-full-%j.log','--export=ALL,SEARCH_G1_BASE='+base+',SEARCH_G1_SOURCE_COMMIT='+commit,base+'/repo/reproduction/common8/launch_native.sh']
code='import json,os,subprocess,time,urllib.request;from pathlib import Path\nb=Path('+repr(base)+');args='+repr(args)+';h='+repr(h)+'\n'+r'''
assert not (b/'logs/run-manifest.json').exists()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for n,size in h['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:19874/'+n,method='HEAD'),timeout=10) as r:assert r.status==200 and int(r.headers['Content-Length'])==size
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(source_commit=args[-2],unix=time.time()),f);f.flush();os.fsync(f.fileno())
(b/'preparation/retrieval-source-handle.json').write_text(json.dumps(h,indent=2))
r=subprocess.run(args,capture_output=True,text=True);proof=dict(argv=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time())
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof));assert r.returncode==0,r.stderr
'''
p=remote('h100-8',code);p.update(base=base,source_commit=commit,retrieval_handle=h);(prep/'attempt2-native-launch.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))

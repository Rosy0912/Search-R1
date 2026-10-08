"""Deploy one immutable source snapshot and submit exactly one dependent native job."""
from pathlib import Path
import sys,subprocess,shlex,json,hashlib,time,base64,re
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
repo=Path('/Users/shaoshitong/Documents/Search-G1/repo');prep=repo.parent/'preparation'
base='/dataset/shaoshitong/Search-G1-common8'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
archive=subprocess.check_output(['git','archive','HEAD'],cwd=repo)
cmd='umask 077; mkdir '+shlex.quote(base+'/repo')+' && tar -xf - -C '+shlex.quote(base+'/repo')
r=subprocess.run(control.route('h100-8',cmd),input=archive,capture_output=True,timeout=55);assert r.returncode==0,r.stderr.decode()
files=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
expected={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in files if Path(n).suffix in ('.py','.sh','.json','.yaml','.yml') and 'evidence' not in Path(n).parts}
code="""import hashlib,json,time;from pathlib import Path
root=Path(BASE)/'repo';expected=json.loads(EXPECTED)
actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json','.yaml','.yml') and 'evidence' not in p.parts}
assert actual==expected
proof=dict(source_commit=COMMIT,source_sha256=actual,accepted=True,checked_unix=time.time(),scope='deployed disk snapshot; not loaded proof or trainer execution')
with (root.parent/'preparation/production-source-deployment.json').open('x') as f:json.dump(proof,f,indent=2)
print(json.dumps(proof))
"""
code=re.sub(r'\b(BASE|EXPECTED|COMMIT)\b',lambda m:repr({'BASE':base,'EXPECTED':json.dumps(expected),'COMMIT':commit}[m.group()]),code)
compile(code,'remote-deployment','exec')
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
(prep/'production-source-deployment.json').write_text(r.stdout)
# Source service is tiny/CPU-only; the large stream is consumed only by allocated target.
source=(repo/'reproduction/common8/assets_export.py').read_bytes()
code="""import base64,json,os,subprocess,time;from pathlib import Path
base=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export');base.mkdir(exist_ok=False)
script=base/'serve_original_retrieval.py';script.write_bytes(base64.b64decode(SOURCE))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (base/'server.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','python3',str(script)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for i in range(100):
 if (base/'handle.json').exists():break
 assert p.poll() is None, 'Source server failed; inspect retained own log'
 time.sleep(.1)
handle=json.loads((base/'handle.json').read_text());assert handle['pid']==p.pid and handle['sid']==p.pid
text=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split();assert int(text[19])==handle['start_ticks']
print(json.dumps(handle))
""".replace('SOURCE',repr(base64.b64encode(source).decode()))
r=subprocess.run(control.route('h100-9','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
handle=json.loads(r.stdout);handle['source_sha256']=hashlib.sha256(source).hexdigest();handle['source_commit']=commit
(prep/'original-retrieval-export-handle.json').write_text(json.dumps(handle,indent=2)+'\n')
code="""import os,json,subprocess,time;from pathlib import Path
base=Path(BASE);claim=base/'preparation/full-native-submit-claim.json'
with claim.open('x') as f:json.dump(dict(submitted_unix=time.time(),source_commit=COMMIT,dependency='afterok:3235'),f);f.flush();os.fsync(f.fileno())
assert not (base/'logs/run-manifest.json').exists()
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=3-00:00:00','--dependency=afterok:3235','--job-name=search-g1-common8-full','--chdir='+str(base/'repo'),'--output='+str(base/'logs/native-full-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(base)+',SEARCH_G1_SOURCE_COMMIT='+COMMIT,str(base/'repo/reproduction/common8/launch_native.sh')]
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(source_commit=COMMIT,argv=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time())
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (base/'preparation/full-native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof));assert r.returncode==0
"""
code=re.sub(r'\b(BASE|COMMIT)\b',lambda m:repr({'BASE':base,'COMMIT':commit}[m.group()]),code)
compile(code,'remote-submit','exec')
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
proof=json.loads(r.stdout);(prep/'full-native-launch.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))

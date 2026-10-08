"""One source-bound discovery child, separate from the still-fresh300 parent.

Large assets travel only H1009 -> H1008 inside the actual native allocation.
An unknown result is observed via the saved intent/remote claims, never retried.
"""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO, PREP = ROOT / 'repo', ROOT / 'preparation'
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control
sys.path.insert(0, str(REPO))
from reproduction.common8.source_files import runtime_files


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        __import__('os').fsync(stream.fileno())


def remote(host, code, payload=None, timeout=90):
    compile(code, '<remote>', 'exec')
    result = subprocess.run(control.route(host, 'python3 -c ' + shlex.quote(code)),
                            input=None if payload is None else json.dumps(payload),
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Observe the saved intent and this same remote namespace before any retry: ' + result.stderr[-2500:])
    return json.loads(result.stdout)


def main():
    intent_path = PREP / 'policy-discovery-native-submit-intent.json'
    assert not intent_path.exists(), 'An attempt was already reserved; observe its remote claims/handle'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=REPO, text=True).strip()
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0'))
    files = {str(p.relative_to(REPO)): p.read_bytes() for p in runtime_files(REPO)}
    assert set(files) <= tracked, 'All deployed runtime files must be in the committed snapshot'
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    test = json.loads((PREP / 'calibration-phase-target-cpu-v3-result.json').read_text())
    summary = json.loads(test['stdout'])
    assert test['returncode'] == test['shell_syntax_returncode'] == 0
    assert summary['successful'] and not summary['skipped'] and not summary['cuda_available']
    assert hashes == test['source_sha256'], 'Source changed after actual target regressions'
    token = uuid.uuid4().hex[:12]
    parent = '/mnt/data/shaoshitong/Search-G1-common8-production-' + token
    base = '/mnt/data/shaoshitong/Search-G1-common8-policy-discovery-' + token
    export = '/dataset/shaoshitong/g1-common8-original-retrieval-export-discovery-' + token
    intent = dict(base=base, parent_base=parent, export_base=export, source_commit=commit,
                  source_sha256=hashes, created_unix=time.time(),
                  scope='Independent one-outer discovery; full fresh300 production remains unstarted')
    save(intent_path, intent)
    payload = dict(intent, files={name: base64.b64encode(data).decode() for name, data in files.items()})
    prepared = remote('h100-8', r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);old=Path('/dataset/shaoshitong/Search-G1-common8-attempt2-39e7968c7831')
for directory in (p['parent_base'],p['base']):
 b=Path(directory);assert not b.exists();b.mkdir(parents=True,exist_ok=False)
 for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
 for name in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):(b/name).symlink_to(old/name,target_is_directory=True)
 for name,data in p['files'].items():
  relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
  target=b/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
  assert hashlib.sha256(target.read_bytes()).hexdigest()==p['source_sha256'][name]
 for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
  source=old/'data'/name;target=b/'data'/name;shutil.copy2(source,target)
  assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
 (b/'preparation/official-model-manifest-original.json').write_bytes((old/'preparation/official-model-manifest-original.json').read_bytes())
 (b/'preparation/source-package-intent.json').write_text(json.dumps({k:v for k,v in p.items() if k!='files'},indent=2))
b=Path(p['base']);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b/'repo'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
code='import verl,verl.single_controller;from verl.workers.fsdp_workers import ActorRolloutRefWorker;from reproduction.common8 import prepare_calibration_run,calibration_runtime,adopt_initial_probes,production_safety_admission;from reproduction.common8.verify_data import verify;import sys,json;print(json.dumps(verify(__import__("pathlib").Path(sys.argv[1]))))'
r=subprocess.run([str(b/'env/bin/python'),'-c',code,str(b/'data')],cwd=b/'repo',env=env,capture_output=True,text=True,timeout=70)
proof=dict(base=str(b),parent_base=p['parent_base'],checked_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_commit=p['source_commit'],source_sha256=p['source_sha256'],scope='CPU import and original canonical-byte verification only; no model, GPU or healthy acceptance')
(b/'preparation/package-cpu-proof.json').write_text(json.dumps(proof,indent=2));assert r.returncode==0,r.stderr
syntax=subprocess.run(['bash','-n',str(b/'repo/reproduction/common8/launch_calibration_native.sh'),str(b/'repo/reproduction/common8/train.sh')],capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
print(json.dumps(proof))
''', payload)
    save(PREP / 'policy-discovery-native-package-proof.json', prepared)
    source = files['reproduction/common8/assets_export.py'].decode().replace(
        "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')", 'BASE=Path(' + repr(export) + ')')
    export_intent = dict(export=export, source=base64.b64encode(source.encode()).decode(),
                         source_sha256=hashlib.sha256(source.encode()).hexdigest())
    handle = remote('h100-9', r'''
import base64,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['export']);assert not b.exists()
with socket.socket() as sock:sock.bind(('10.33.0.13',19874))
b.mkdir(exist_ok=False);s=b/'serve_original_retrieval.py';s.write_bytes(base64.b64decode(p['source']))
assert hashlib.sha256(s.read_bytes()).hexdigest()==p['source_sha256']
with (b/'start-claim.json').open('x') as f:json.dump(dict(export=str(b),source_sha256=p['source_sha256'],created_unix=time.time()),f);f.flush();os.fsync(f.fileno())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (b/'server.log').open('x') as f:process=subprocess.Popen(['nice','-n','19','python3',str(s)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text().rsplit(')',1)[1].split()
with (b/'launcher-process.json').open('x') as f:json.dump(dict(pid=process.pid,start_ticks=int(stat[19]),sid=int(stat[3]),uid=os.getuid(),source_sha256=p['source_sha256']),f);f.flush();os.fsync(f.fileno())
h=None
for _ in range(100):
 assert process.poll() is None
 try:
  h=json.loads((b/'handle.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
assert h is not None and h['pid']==process.pid
h.update(export=str(b),source_sha256=p['source_sha256']);print(json.dumps(h))
''', export_intent)
    save(PREP / 'policy-discovery-retrieval-export-handle.json', handle)
    payload = dict(intent, handle=handle)
    launched = remote('h100-8', r'''
import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base'])
for namespace in (b,parent):
 assert not (namespace/'logs/run-manifest.json').exists()
 for name,sha in p['source_sha256'].items():assert hashlib.sha256((namespace/'repo'/name).read_bytes()).hexdigest()==sha
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,size in p['handle']['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:19874/'+name,method='HEAD'),timeout=10) as response:assert response.status==200 and int(response.headers['Content-Length'])==size
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name=search-g1-policy-discovery','--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-discovery-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+str(parent)+',SEARCH_G1_CALIBRATION_ROLE=policy_safety_calibration_discovery,SEARCH_G1_SOURCE_COMMIT='+p['source_commit'],str(b/'repo/reproduction/common8/launch_calibration_native.sh')]
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(argv=args,created_unix=time.time(),base=str(b),parent_base=str(parent),scope='Independent discovery child, no full300 producer started'),f);f.flush();os.fsync(f.fileno())
(b/'preparation/retrieval-source-handle.json').write_text(json.dumps(p['handle'],indent=2))
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(base=str(b),parent_base=str(parent),source_commit=p['source_commit'],source_sha256=p['source_sha256'],argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_unix=time.time(),actual_production_updates=0)
if r.returncode==0:
 proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr;print(json.dumps(proof))
''', payload)
    save(PREP / 'policy-discovery-native-launch.json', launched)
    print(json.dumps({k: v for k, v in launched.items() if k not in ('source_sha256', 'argv')}, indent=2))


if __name__ == '__main__':
    main()

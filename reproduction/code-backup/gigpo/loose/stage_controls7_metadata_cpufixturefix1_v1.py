"""Exclusive owned metadata and unexecuted source transfer, not a deployment."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006';d=base/'work/gigpo-e1-quality-resume-runtime/fresh200-deployment-controls-v7'
f=base/'outputs/gigpo-e1-policy-safety/FRESH200-DEPLOYMENT-CONTROLS-STATIC-MANIFEST-v7-cpufixturefix1.json';m=json.loads(f.read_text());items=[]
def add(local,target):
 raw=Path(local).read_bytes();items.append(dict(path=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode()))
for r in m['files']:add(d/Path(r['path']).name,root+'/tools/fresh200-deployment-controls-v7/'+Path(r['path']).name)
add(d/'controls_cpu_once_v7.py',root+'/tools/management/controls-cpu-once-v7.py')
for p in [f,base/'outputs/gigpo-e1-policy-safety/FRESH200-DEPLOYMENT-CONTROLS-FREEZE-v7-cpufixturefix1.json',base/'outputs/gigpo-e1-policy-safety/FRESH200-DEPLOYMENT-CONTROLS-AUTHOR-STOP-v7-cpufixturefix1.json',base/'outputs/gigpo-e1-policy-safety/FRESH200-CONTROLS-v7-REQUIRED-PROOF-PROTOCOL-cpufixturefix1-v1.md',base/'outputs/gigpo-e1-policy-safety/FRESH200-DEPLOYMENT-CONTROLS-STATIC-HANDOFF-v7-cpufixturefix1.md',base/'work/gigpo-e1-quality-resume-runtime/fresh200-deployment-v1/EXPECTED-READONLY-ASSET-SHA-CLOSURE-v1.json',base/'outputs/gigpo-e1-policy-safety/policy-safety-calibration-52befe9-physical-v9.json',base/'outputs/gigpo-e1-policy-safety/action-prefix-native-CANDIDATE-2262942-v1.json']:add(p,root+'/logs/'+p.name)
code='''import base64,datetime,hashlib,json,os,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root']);records=[]
for row in p['items']:
 path=Path(row['path']);assert path.is_relative_to(root) and not path.is_symlink();path.parent.mkdir(parents=True,exist_ok=True);assert path.parent.resolve()==path.parent
 raw=base64.b64decode(row['b64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 data=path.read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];records.append({key:row[key] for key in ('path','bytes','sha256')})
directory=root/'tools/fresh200-deployment-controls-v7'
assert sorted(str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file())==sorted(p['six_names'])
assert all(path.is_file() and not path.is_symlink() for path in directory.iterdir())
print(json.dumps(dict(status='EXACT_CONTROLS7_AND_ORIGINAL_PROOFS_METADATA_ONLY_NOT_EXECUTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_original_records=records,controls_directory=str(directory),actual_six_names=sorted(p['six_names']),separate_management_helper=True,CPU_executed=False,GPU_executed=False,READY=False,submitted=False,formal_iterations=0),indent=2))
'''
inner='/usr/bin/python3 -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
payload=dict(root=root,items=items,six_names=[Path(r['path']).name for r in m['files']]);r=subprocess.run(route,input=json.dumps(payload).encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
x=json.loads(r.stdout);out=base/'outputs/gigpo-e1-instruct200/FRESH200-CONTROLS7-STAGING-METADATA-RECEIPT-v1.json'
with out.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(dict(status=x['status'],UTC=x['UTC'],record_count=len(x['metadata_original_records']),receipt=str(out),CPU_executed=False,GPU_executed=False,submitted=False)))

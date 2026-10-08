"""Exclusive owned metadata and unexecuted source transfer, not a deployment."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006';items=[]
def add(local,target):
 raw=Path(local).read_bytes();items.append(dict(path=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode()))
stop_path=base/'work/gigpo-e1-quality-resume-runtime/fresh200-live-deployment-inputs-v8-prephysical-v1/AUTHOR-STOP.json'
stop_raw=stop_path.read_bytes();assert len(stop_raw)==4155 and hashlib.sha256(stop_raw).hexdigest()=='856fd2c5db6a93cee068f4a1b9b709a8ace4f35563cb60b3b6970f8833502d11'
stop=json.loads(stop_raw);assert stop['all_author_metadata_writes_stopped'] is True
for record in stop['frozen_bundle_records']:
 p=Path(record['path']);raw=p.read_bytes();assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
 name=p.name if p.name=='FRESH200-LIVE200-INPUTS-v8-PREPHYSICAL-v1.json' else 'FRESH200-V8-PREPHYSICAL-'+p.name
 add(p,root+'/logs/'+name)
add(stop_path,root+'/logs/FRESH200-V8-PREPHYSICAL-AUTHOR-STOP.json')
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
print(json.dumps(dict(status='EXACT_FROZEN_V8_PREPHYSICAL_INPUTS_METADATA_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_original_records=records,separate_management_helper=True,CPU_executed=False,GPU_executed=False,READY=False,submitted=False,formal_iterations=0),indent=2))
'''
inner='/usr/bin/python3 -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
payload=dict(root=root,items=items);r=subprocess.run(route,input=json.dumps(payload).encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
x=json.loads(r.stdout);out=base/'outputs/gigpo-e1-instruct200/FRESH200-V8-PREPHYSICAL-INPUTS-METADATA-STAGING-RECEIPT-v1.json'
with out.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(dict(status=x['status'],UTC=x['UTC'],record_count=len(x['metadata_original_records']),receipt=str(out),CPU_executed=False,GPU_executed=False,submitted=False)))

"""Exclusive owned metadata and unexecuted source transfer, not a deployment."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006';items=[]
def add(local,target):
 raw=Path(local).read_bytes();items.append(dict(path=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode()))
add(base/'work/controls_cpu_once_v8_depaliasfix1.py',root+'/tools/management/controls-cpu-once-v8.py')
for name in ['FRESH200-DEPLOYMENT-CONTROLS-STATIC-MANIFEST-v7-depaliasfix1-final.json']:
 p=base/'outputs/gigpo-e1-policy-safety'/name;add(p,root+'/logs/'+name)
for name in ['CONTROLS7-ACTUAL-DEPENDENCY-ALIAS-READONLY-v1.json','CONTROLS7-HELPER8-DEPALIASFIX1-AUTHOR-STOP-v1.json','SOURCE11-CPUFIX1-329-FACTUAL-CPU-RECEIPT-v1.json','RUNTIME12-CPUFIX1-204-FACTUAL-CPU-RECEIPT-v1.json','CONTINUOUS-SOURCE11-CPUFIX1-CPU-INDEPENDENT-REVIEW-v1.json','CONTINUOUS-RUNTIME12-CPUFIX1-CPU-INDEPENDENT-REVIEW-v1.json']:
 p=base/'outputs/gigpo-e1-instruct200'/name;add(p,root+'/logs/'+name)
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
print(json.dumps(dict(status='EXACT_UNEXECUTED_HELPER8_AND_ORIGINAL_ACCEPTED_CPU_PROOFS_METADATA_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_original_records=records,separate_management_helper=True,CPU_executed=False,GPU_executed=False,READY=False,submitted=False,formal_iterations=0),indent=2))
'''
inner='/usr/bin/python3 -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
payload=dict(root=root,items=items);r=subprocess.run(route,input=json.dumps(payload).encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
x=json.loads(r.stdout);out=base/'outputs/gigpo-e1-instruct200/FRESH200-HELPER8-AND-ACCEPTED-CPU-METADATA-STAGING-RECEIPT-v1.json'
with out.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(dict(status=x['status'],UTC=x['UTC'],record_count=len(x['metadata_original_records']),receipt=str(out),CPU_executed=False,GPU_executed=False,submitted=False)))

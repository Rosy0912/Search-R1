"""Exclusive owned metadata and unexecuted source transfer, not a deployment."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006';items=[]
def add(local,target):
 raw=Path(local).read_bytes();items.append(dict(path=target,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode()))
expected={
 'CONTROLS8-INDEPENDENT-PREEXEC-REVIEW-v1.json':(211339,'795578800f0e57a10ec121225bd15ac035f368616433d57eacc66a59d6b423d7'),
 'CONTROLS8-HELPER9-FACTUAL-CPU-RECEIPT-v1.json':(112022,'0f93760ba69a6cc579259431aa6479ee00a92b5d899e23300e8303499119dfa8'),
 'CONTROLS8-HELPER9-CPU-INDEPENDENT-REVIEW-v1.json':(166540,'415b0b2b6ab79637cce061aa8ca7c1916cc0c0ee2e7120c8c038f009062f7d63'),
 'CONTROLS8-HELPER9-CPU-INDEPENDENT-REVIEW-v1.md':(3646,'2918650e568f23eb67e55a784824af5455bbf32de039e4c8aef009db73e02ee1')}
for name,(size,digest) in expected.items():
 p=base/'outputs/gigpo-e1-instruct200'/name;raw=p.read_bytes();assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest;add(p,root+'/logs/'+name)
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
print(json.dumps(dict(status='EXACT_CONTROLS8_TRUE_CPU_PROOFS_METADATA_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_original_records=records,separate_management_helper=True,CPU_executed=False,GPU_executed=False,READY=False,submitted=False,formal_iterations=0),indent=2))
'''
inner='/usr/bin/python3 -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
payload=dict(root=root,items=items);r=subprocess.run(route,input=json.dumps(payload).encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
x=json.loads(r.stdout);out=base/'outputs/gigpo-e1-instruct200/FRESH200-CONTROLS8-ACCEPTED-CPU-METADATA-STAGING-RECEIPT-v1.json'
with out.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(dict(status=x['status'],UTC=x['UTC'],record_count=len(x['metadata_original_records']),receipt=str(out),CPU_executed=False,GPU_executed=False,submitted=False)))

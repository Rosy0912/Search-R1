import subprocess,hashlib,json,os,sys,shlex
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/probability-diagnostic-3307-v2-20261006')
if sys.flags.optimize or os.uname().nodename!='h100-16': raise RuntimeError('exact head required')
if root.resolve()!=root or (root/'logs').resolve()!=root/'logs':raise RuntimeError('canonical head metadata required')
remote_check="from pathlib import Path; import json; p=Path("+repr(str(root/'scheduler-head'))+"); print(json.dumps([x.name for x in p.iterdir() if x.name.endswith(('submit-intent.json','receipt.json','worker-intent.json'))]))"
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','h100-5','python3 -B -c '+shlex.quote(remote_check)],check=True,capture_output=True)
if json.loads(r.stdout):raise RuntimeError('existing worker transaction requires inspection')
result=subprocess.run(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','h100-5','cat '+str(root/'logs/diagnostic-approval.json')],check=True,capture_output=True)
raw=result.stdout
if hashlib.sha256(raw).hexdigest()!='7e2d1a743a786a265c1004cec4f237d81fe3171dd2184df9a069e017266b369a': raise RuntimeError('immutable worker approval SHA mismatch')
a=json.loads(raw)
expected={'root':str(root),'source_commit':'aec5e057093b4c491bfb0a47abd7605d98981b56','tools_manifest_sha256':'3d6c8a8593ca5f4cf9e9f79294fae3042f38433f6aeb90cc81bf860059596fa9','nonce':'5e1669e5ef1fd15fb01093604ac7efe3','optimizer_writes':0,'scheduler_updates':0,'formal_outer_updates':0,'production_release':False,'native_acceptance':False}
if a.get('protocol')!='BAPO_ZERO_ADAM_DIAGNOSTIC_V1' or a.get('status')!='ALLOW_ZERO_ADAM_PROBABILITY_REPLAY' or type(a.get('allocation_gpu_count')) is not int or a['allocation_gpu_count']!=8 or type(a.get('probe_gpu_count')) is not int or a['probe_gpu_count']!=1 or any(type(a.get(k)) is not int for k in ('optimizer_writes','scheduler_updates','formal_outer_updates')):raise RuntimeError('scope/type mismatch')
if any(a.get(k)!=v for k,v in expected.items()):raise RuntimeError('approval binding mismatch')
if list((root/'scheduler-head').glob('*submit-intent.json')) or list((root/'scheduler-head').glob('*receipt.json')) or list((root/'scheduler-head').glob('*worker-intent.json')):raise RuntimeError('previous submit outcome requires inspection')
with (root/'logs/diagnostic-approval.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
fd=os.open(str(root/'logs'),os.O_DIRECTORY);os.fsync(fd);os.close(fd)
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','h100-5','python3 -B -c '+shlex.quote(remote_check)],check=True,capture_output=True)
if json.loads(r.stdout):raise RuntimeError('worker transaction appeared during mirror')
print(json.dumps({'scope':'EXACT_EXISTING_WORKER_APPROVAL_HEAD_METADATA_MIRROR','sha256':hashlib.sha256(raw).hexdigest(),'nonce':a['nonce'],'job_submitted':False}))

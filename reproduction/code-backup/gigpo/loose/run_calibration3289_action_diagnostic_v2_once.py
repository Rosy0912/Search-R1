"""Run one reviewed CPU-only diagnostic; never submit Slurm or use a model."""
import hashlib
import json
import subprocess
from pathlib import Path

base = Path('outputs/gigpo-e1-policy-safety')
review_path = base / 'CALIBRATION3289-ACTION-DIAGNOSTIC-INDEPENDENT-REVIEW-v2.json'
review_raw = review_path.read_bytes()
review = json.loads(review_raw)
assert review['status']=='PASS' and review['scope']=='H100_CPU_READONLY_DIAGNOSTIC_TOOL_STATIC_ONLY_NOT_ACTUAL_PROBABILITY_ACCEPTANCE'
handoff = json.loads((base/'CALIBRATION3289-ACTION-DIAGNOSTIC-REVIEW-HANDOFF-v2.json').read_text())
for key in ['implementation', 'identity_input', 'frozen_failure_inventory']:
    row=handoff[key]
    raw=Path(row['path']).read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004'
helper=root+'/tools/diagnostic-action-3289-v2/diagnose_actions_v1.py'
identity=root+'/logs/CALIBRATION3289-ACTION-IDENTITY-CPU-INPUT-v2.json'
inventory=root+'/logs/CALIBRATION3289-FAILURE-PAYLOAD-v1.json'
output=root+'/logs/CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-v2.json'
implementation=json.loads(Path(handoff['implementation']['path']).read_text())
helper_row=next(r for r in implementation['files'] if r['path'].endswith('/diagnose_actions_v1.py'))
assert review['implementation_sha256']==handoff['implementation']['sha256']
assert review['identity_input_sha256']==handoff['identity_input']['sha256']
assert review['inventory_sha256']==handoff['frozen_failure_inventory']['sha256']
assert review['diagnostic_helper_sha256']==helper_row['sha256']
remote_code=r'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(ROOT_VALUE)
helper=Path(HELPER_VALUE)
identity=Path(IDENTITY_VALUE)
inventory=Path(INVENTORY_VALUE)
output=Path(OUTPUT_VALUE)
claim=root/'logs/CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-CLAIM-v2.json'
receipt=root/'logs/CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-EXECUTION-v2.json'
stdout_path=root/'logs/CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-STDOUT-v2.log'
stderr_path=root/'logs/CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-STDERR-v2.log'
for p in [output,claim,receipt,stdout_path,stderr_path]:
 assert not os.path.lexists(p),'Existing diagnostic artifact; inspect instead of repeating: '+str(p)
for p,expected in [(helper,HELPER_SHA_VALUE),(identity,IDENTITY_SHA_VALUE),(inventory,INVENTORY_SHA_VALUE)]:
 assert p.is_file() and not p.is_symlink() and p.resolve()==p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
def save(p,d):
 with p.open('x') as f:
  json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
save(claim,{'scope':'ONCE_CPU_ONLY_NOT_TRAINING','wrapper_pid':os.getpid(),'started_unix_ns':time.time_ns(),
 'review_sha256':REVIEW_SHA_VALUE,'helper_sha256':HELPER_SHA_VALUE,'identity_sha256':IDENTITY_SHA_VALUE,
 'inventory_sha256':INVENTORY_SHA_VALUE,'output':str(output)})
env=os.environ.copy()
for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH']:
 env.pop(name,None)
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(root/'repo-policy-safety-52befe9'),
 NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1')
argv=[str(root/'env/bin/python'),str(helper),'--identity',str(identity),
 '--identity-sha256',IDENTITY_SHA_VALUE,'--inventory',str(inventory),'--output',str(output)]
with stdout_path.open('xb') as out,stderr_path.open('xb') as err:
 process=subprocess.Popen(argv,env=env,stdout=out,stderr=err)
 exit_code=process.wait()
def record(p):
 raw=p.read_bytes()
 return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
result={'scope':'ACTUAL_CPU_DIAGNOSTIC_EXECUTION_NOT_PROBABILITY_ACCEPTANCE',
 'job_id':3289,'node':'h100-8','child_pid':process.pid,'exit_code':exit_code,
 'finished_unix_ns':time.time_ns(),'claim':record(claim),'stdout':record(stdout_path),'stderr':record(stderr_path),
 'cuda_visible_devices':'','model_or_optimizer_requested':False,'formal_iterations_contributed':0}
if output.exists():
 d=json.loads(output.read_text());result['output']=record(output);result['diagnostic_status']=d['status']
save(receipt,result)
print(json.dumps({'execution_receipt':record(receipt),'actual':result},ensure_ascii=False))
sys.exit(exit_code)
'''
replacements={
 'ROOT_VALUE':repr(root),'HELPER_VALUE':repr(helper),'IDENTITY_VALUE':repr(identity),
 'INVENTORY_VALUE':repr(inventory),'OUTPUT_VALUE':repr(output),
 'HELPER_SHA_VALUE':repr(helper_row['sha256']),
 'IDENTITY_SHA_VALUE':repr(handoff['identity_input']['sha256']),
 'INVENTORY_SHA_VALUE':repr(handoff['frozen_failure_inventory']['sha256']),
 'REVIEW_SHA_VALUE':repr(hashlib.sha256(review_raw).hexdigest()),
}
for key,value in replacements.items():remote_code=remote_code.replace(key,value)
compile(remote_code,'cpu-diagnostic-execution','exec')
script='/usr/bin/python3 - <<\'REMOTE\'\n'+remote_code+'\nREMOTE\n'
r=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True)
raise SystemExit(r.returncode)

import json,pathlib,subprocess
remote=r'''python3 - <<'INNER'
import base64,hashlib,json,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');old=r/'logs/qwen25-32b-terminal-fixed413-eval64-3023'
evidence={}
for name in ('evaluation.log','supervisor-result.json','final-gpu.json','baseline.json','baseline-nvml.json'):
 b=(old/name).read_bytes();evidence[name]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
result=json.loads((old/'supervisor-result.json').read_text())
assert result['cleanup']['actual_cgroup_empty'] and not result['cleanup']['remaining'] and not result['foreign_baseline_missing']
lookup=subprocess.run(['scontrol','show','job','3023'],text=True,capture_output=True)
job=lookup.stdout
if lookup.returncode==0:assert 'JobState=FAILED' in job,job
else:
 assert 'Invalid job id' in lookup.stderr,lookup.stderr
 assert result['status']=='FAIL_NO_TRAINING',result['status']
 job='AGED_OUT_NO_ACCOUNTING: '+lookup.stderr
def collect_pids(obj):
 if isinstance(obj,dict):
  if 'pid' in obj:yield int(obj['pid'])
  for v in obj.values():yield from collect_pids(v)
 elif isinstance(obj,list):
  for v in obj:yield from collect_pids(v)
owned_pids=sorted(set(collect_pids(result['cleanup'])))
assert all(not pathlib.Path('/proc',str(pid)).exists() for pid in owned_pids),owned_pids
group=pathlib.Path('/sys/fs/cgroup/freezer'+result['cgroup'])
if group.exists():
 assert all(not p.read_text().strip() for p in group.rglob('cgroup.procs'))
queue=subprocess.run(['squeue','-h','-j','3023'],text=True,capture_output=True)
assert not queue.stdout.strip(),queue.stdout
record=r/'logs/qwen25-32b-eval64-spawn-v1-submission.json'
entry=r/'tools/qwen25-32b-eval64-spawn-v1.sbatch';sha='b55b536e2c2294896459195f0bbef038b6820be4f4f81a78f23bc4d12335b658'
assert hashlib.sha256(entry.read_bytes()).hexdigest()==sha
binding=r/'logs/qwen25-32b-terminal-fixed413-binding.json';assert hashlib.sha256(binding.read_bytes()).hexdigest()=='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
for row in gpu.splitlines():
 uuid,used,total,sm=[x.strip() for x in row.split(',')];assert float(used)/float(total)<=.15 and float(sm)<=15,gpu
if record.exists():
 state=json.loads(record.read_text());assert state['status']=='SUBMITTED',state
else:
 active=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True);assert not any(x.split('|',1)[-1]=='qwen25-32b-eval64-spawn-v1' for x in active.splitlines())
 state=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(entry),entry_sha256=sha,binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),epoch=time.time(),prior_failed_job=3023,prior_job_record=job,actual_owned_pids_absent=owned_pids,actual_old_cgroup_empty_or_gone=True,pre_submit_gpu=gpu,change='VLLM_WORKER_MULTIPROC_METHOD=spawn; fixed source/tools/same64 unchanged')
 record.write_text(json.dumps(state,indent=2)+'\n')
 out=subprocess.check_output(['sbatch','--parsable',str(entry),str(binding)],text=True).strip();jid=out.split(';')[0];assert jid.isdigit(),out
 state.update(status='SUBMITTED',job_id=int(jid));record.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'submission':state,'failed_evidence':evidence}))
INNER'''
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
if p.returncode:print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
data=json.loads(p.stdout);d=pathlib.Path('outputs/remote-evidence/eval64-3023-failed');d.mkdir(parents=True,exist_ok=True)
import base64,hashlib
for name,rec in data['failed_evidence'].items():
 b=base64.b64decode(rec['base64']);assert hashlib.sha256(b).hexdigest()==rec['sha256'];(d/name).write_bytes(b)
pathlib.Path('outputs/qwen25-32b-eval64-spawn-v1-submission.json').write_text(json.dumps(data['submission'],indent=2)+'\n')
print(json.dumps(data['submission']))

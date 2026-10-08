from pathlib import Path
import json,hashlib,time,subprocess,fcntl,math
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');source=root/'runs/author-native-continue-300-u';stage=root/'continue-300-tools-w';claim=root/'author-native-continue-300-w-source-handoff.json'
lock=(root/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not claim.exists() and not (root/'author-native-continue-300-w-launch-claim.json').exists()
gate=json.loads((stage/'cpu-original-fallback-gate.json').read_text());assert gate['passed'] and not gate['CUDA_initialized'] and gate['current_settings_preserved'] and gate['bounded_two_attempts']
assert json.loads((stage/'cpu-bounded-gate.json').read_text())['passed']
job=str(json.loads((source/'launch.json').read_text())['job']);assert job=='3201'
q=subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).split();assert not q,q
vroot=root/'runs/author-native-continue-300-v'
assert json.loads((vroot/'driver-exit.json').read_text())['returncode']==1
for vf in (vroot/'evidence/workers').glob('*.jsonl'):
 for line in vf.read_text().splitlines():
  va=json.loads(line)
  if va.get('kind')=='ack':
   vp=Path('/proc')/str(va['pid'])
   if vp.exists():assert int((vp/'stat').read_text().rsplit(')',1)[1].split()[19])!=va['process_start_ticks']

assert json.loads((source/'driver-exit.json').read_text())['returncode']==1
rows=[]
for p in list((source/'evidence').glob('controller-*.jsonl'))+list((source/'evidence/workers').glob('*.jsonl')):
 b=p.read_bytes();rows.extend(json.loads(l) for l in b[:b.rfind(b'\n')+1].splitlines())
acks=[v for v in rows if v.get('kind')=='ack'];assert len(acks)==8 and all(str(v['job'])==job for v in acks)
record=dict(status='failed_source_checkpoint_verification',source_run=source.name,source_job=job,created_at=time.time(),before_stop_completed_outers=[v['step'] for v in rows if v.get('kind')=='outer'],uncommitted_inflight_compute_may_be_discarded=True)
claim.write_text(json.dumps(record,indent=2)+'\n')
assert not q  # Already failed source; no cancellation
def original_pid_alive(a):
 p=Path('/proc')/str(a['pid'])
 try:return int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==a['process_start_ticks']
 except FileNotFoundError:return False
for _ in range(90):
 q=job in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
 if not q and not any(original_pid_alive(a) for a in acks):break
 time.sleep(1)
else:raise RuntimeError('Source resources not released; no new launch')
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
rows=[];prefixes=[]
for p in list((source/'evidence').glob('controller-*.jsonl'))+list((source/'evidence/workers').glob('*.jsonl')):
 b=p.read_bytes();b=b[:b.rfind(b'\n')+1];prefixes.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));rows.extend(json.loads(l) for l in b.splitlines())
expected={(role,rank) for role in ['main','reranker'] for rank in range(4)}
def finite(v):
 if isinstance(v,dict):return all(finite(z) for z in v.values())
 if isinstance(v,list):return all(finite(z) for z in v)
 return not isinstance(v,(int,float)) or math.isfinite(v)
selected=None
for cp in sorted((source/'checkpoints').glob('global_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1]),reverse=True):
 step=int(cp.name.rsplit('_',1)[1]);updates=[v for v in rows if v.get('kind')=='update' and v.get('outer_step')==step];ends=[v for v in rows if v.get('kind')=='global_update_completed' and v.get('step')==step]
 if len(updates)!=8 or {(v['role'],v['rank']) for v in updates}!=expected or {v['role'] for v in ends}!={'main','reranker'}:continue
 if not all(v['actual_adam_delta']>0 and finite(v['original_metrics']) for v in updates):continue
 sides=[]
 try:
  assert (cp/'data.pt').is_file()
  for role in ['actor','reranker_actor_rollout']:
   for rank in range(4):
    p=cp/role/f'native-integrity-rank-{rank}.json';x=json.loads(p.read_text());assert str(x['source_job'])==job and x['rank']==rank and x['path']==str(p.parent)
    saved=[v for v in rows if v.get('kind')=='checkpoint_saved_full_digest' and v.get('path')==x['path'] and v.get('rank')==rank];assert len(saved)==1 and saved[0]['state']==x['state']
    assert x['state']['master_dtypes']==['torch.float32'] and x['state']['optimizer']['tensor_bytes']>0 and len(x['files'])==3
    sides.append((p,x))
 except (AssertionError,FileNotFoundError):continue
 selected=(cp,step,sides);break
assert selected is not None,'No complete saved source; do not launch or roll back silently'
cp,step,sides=selected;files=[]
for p,x in sides:
 for name,v in x['files'].items():
  f=p.parent/name;h=hashlib.sha256()
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(8388608),b''):h.update(b)
  assert h.hexdigest()==v['sha256'] and f.stat().st_size==v['bytes'],str(f)
 files.append(dict(role=p.parent.name,rank=x['rank'],source_job=job,integrity_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files=x['files']))
assert len(files)==8 and sum(len(v['files']) for v in files)==24
result=dict(source_run=source.name,source=source.name,source_job=job,source_step=step,checkpoint=str(cp),checked_at=time.time(),ranks=8,files=files,data_sha256=hashlib.sha256((cp/'data.pt').read_bytes()).hexdigest(),data_bytes=(cp/'data.pt').stat().st_size,actual_full_file_shas_reverified=True,source_resources_released=True,source_complete_prefixes=prefixes,controlled_restart=False,source_driver0_required=False)
(stage/'source-checkpoint-verification.json').write_text(json.dumps(result,indent=2)+'\n');record.update(status='failed_source_latest_full_checkpoint_verified',completed_at=time.time(),source_step=step,verification_path=str(stage/'source-checkpoint-verification.json'));claim.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(result),flush=True)

"""Observe original G1 update invocations, never replaces reward/loss/update math."""
import hashlib,json,os,pathlib,time

def atomic(path,record):
 p=pathlib.Path(path)
 with p.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def sampled_parameters(actor):
 import torch
 values=[]
 for p in actor.actor_module.parameters():
  if p.numel():values.append(p.detach().reshape(-1)[:64].float().cpu())
 raw=b''.join(v.numpy().tobytes() for v in values)
 return hashlib.sha256(raw).hexdigest()
def observed_update(actor,original,data):
 import torch
 optimizer_steps=[];micro=[]
 before=sampled_parameters(actor)
 forward=actor._forward_micro_batch;step=actor._optimizer_step
 def observed_forward(*args,**kwargs):
  output=forward(*args,**kwargs);micro.append(dict(epoch=time.time(),finite=all(bool(torch.isfinite(t).all()) for t in output)));return output
 def observed_step(*args,**kwargs):
  began=time.time();lr=[float(group['lr']) for group in actor.actor_optimizer.param_groups];norm=step(*args,**kwargs);optimizer_steps.append(dict(started_epoch=began,completed_epoch=time.time(),grad_norm=float(norm),finite=bool(torch.isfinite(norm)),actual_learning_rates=lr));return norm
 actor._forward_micro_batch=observed_forward;actor._optimizer_step=observed_step
 try:
  need=['old_log_probs','advantages','loss_mask','ref_log_prob']
  finite={n:bool(torch.isfinite(data.batch[n]).all()) for n in need}
  output=original(data)
  return output,dict(local_rows=len(data),input_finite=finite,micro_forward_records=micro,optimizer_step_records=optimizer_steps,parameter_sample_before=before,parameter_sample_after=sampled_parameters(actor),sample_scope='FIRST64_PER_NONEMPTY_LOCAL_PARAMETER_NOT_FULL_MODEL_HASH',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 finally:actor._forward_micro_batch=forward;actor._optimizer_step=step

def verify_updates(records,outer_steps):
 if len(records)!=8 or {r['rank'] for r in records}!=set(range(8)):raise ValueError('Need8 actual rank receipts')
 for r in records:
  updates=r['updates']
  if len(updates)!=outer_steps or any(u['local_rows']!=80 for u in updates):raise ValueError('Actual640/world8 rows missing')
  if sum(len(u['optimizer_step_records']) for u in updates)!=outer_steps*3 or sum(len(u['micro_forward_records']) for u in updates)!=outer_steps*10:raise ValueError('ActualG1 optimizer/micro count mismatch')
  if any(set(u['input_finite'])!={'old_log_probs','advantages','loss_mask','ref_log_prob'} or not all(u['input_finite'].values()) or any(not x['finite'] for x in u['micro_forward_records']) or any(not x['finite'] or x['grad_norm']<=0 for x in u['optimizer_step_records']) for u in updates):raise ValueError('Nonfinite/zero actual update')
 if not any(u['parameter_sample_before']!=u['parameter_sample_after'] for r in records for u in r['updates']):raise ValueError('No observed changed local parameter sample')
 return dict(scope='ACTUAL_G1_RESOURCE_UPDATES_UNREVIEWED_NOT_100_TRAINING',actual_outer_steps=outer_steps,optimizer_steps_per_rank=3*outer_steps,micro_forwards_per_rank=10*outer_steps,ranks=8)

def verify_ready(directory,config,trainer_pid,tools,snapshot_call,validate_peer,prefix=""):
 """Both original groups completed init_model; confirm live same-job GPU identity."""
 directory=pathlib.Path(directory);paths=[directory/(prefix+'%s-ready-rank%d.json'%(role,rank)) for role in ('actor','ref') for rank in range(8)]
 if not all(p.is_file() for p in paths):return None
 records=[];uuids=set(config['uuids'])
 snapshot=snapshot_call()
 for role in ('actor','ref'):
  group=[]
  for rank in range(8):
   path=directory/(prefix+'%s-ready-rank%d.json'%(role,rank))
   if path.is_symlink():raise ValueError('worker ready alias')
   r=json.loads(path.read_text());current=tools.process_identity(r['pid'])
   if prefix and r.get('outer_step')!=int(prefix.removeprefix('round').removesuffix('-')):raise ValueError('actual resume outer identity changed')
   if r['role']!=role or r['rank']!=rank or r['job']!=config['job'] or r['hostname']!='h100-11' or r['cgroup']!=config['cgroup'] or current['starttime_ticks']!=r['starttime_ticks']:raise ValueError('actual worker ready identity changed')
   validate_peer(r['pid'],os.getuid(),trainer_pid,config['cgroup'])
   actual=sorted({a['uuid'] for a in snapshot['applications'] if a['pid']==r['pid']})
   if actual!=r['gpu_uuids'] or len(actual)!=1 or actual[0] not in uuids or r['allocated_bytes']<0 or r['reserved_bytes']<r['allocated_bytes']:raise ValueError('actual worker init GPU/allocator coverage')
   group.append(actual[0]);records.append(dict(r,source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  if set(group)!=uuids:raise ValueError('actual worker role all8 GPU coverage')
 return {'scope':'ACTUAL_WORKER_INIT_READY_NOT_UPDATE_RESOURCE_PROOF','records':records,'nvml_snapshot':snapshot,'completed_epoch':time.time()}


def charged_memory(scope):
 """Actual cgroup counters, separate from sampled summed/shared process RSS."""
 scope=pathlib.Path(scope);values={}
 for name in ('memory.current','memory.peak','memory.max','memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes'):
  p=scope/name
  if p.is_file():values[name]={'path':str(p),'raw':p.read_text().strip()}
 return {'scope':'ACTUAL_CGROUP_CHARGED_MEMORY_COUNTERS_NOT_HOST_AVAILABLE_RAM','cgroup_path':str(scope),'available':bool(values),'values':values}

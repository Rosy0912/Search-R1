"""Observe original Search-R1 update invocations, never replaces reward/loss/update math."""
import hashlib,json,os,pathlib,time,tempfile,math

def atomic(path,record):
 p=pathlib.Path(path);fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.tmp',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(record,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  os.link(tmp,p) # Exclusive publication, never overwrite original receipt.
  directory=os.open(p.parent,os.O_RDONLY)
  try:os.fsync(directory)
  finally:os.close(directory)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def sampled_parameters(actor):
 import torch
 values=[]
 for p in actor.actor_module.parameters():
  if p.numel():values.append(p.detach().reshape(-1)[:64].float().cpu())
 raw=b''.join(v.numpy().tobytes() for v in values)
 return hashlib.sha256(raw).hexdigest()
MEMORY_STAT_KEYS=('allocated_bytes.all.current','allocated_bytes.all.peak','reserved_bytes.all.current','reserved_bytes.all.peak','active_bytes.all.current','active_bytes.all.peak','inactive_split_bytes.all.current','inactive_split_bytes.all.peak','num_alloc_retries','num_ooms')
def pre_backward_memory(actor,torch_module,micro_ordinal,micro_batch=None):
 context=getattr(actor,'_r1_forward_diagnostic_context',None)
 if context is None:return None
 stats=torch_module.cuda.memory_stats()
 selected={key:stats[key] for key in MEMORY_STAT_KEYS}
 if any(type(value) is not int or value<0 for value in selected.values()):raise ValueError('Original native CUDA memory statistics invalid')
 record={key:value for key,value in context.items() if key!='directory'}
 record.update(scope='MICRO_FORWARD_COMPLETED_BEFORE_BACKWARD_NOT_COMPLETED_UPDATE_NOT_OPTIMIZER_STEP',micro_forward_ordinal=micro_ordinal,epoch=time.time(),original_cuda_memory_stats=selected,allocator_environment=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
 if micro_batch is not None and 'input_ids' in micro_batch and 'attention_mask' in micro_batch:
  record['actual_micro_input']={'input_ids_shape':list(micro_batch['input_ids'].shape),'attention_mask_shape':list(micro_batch['attention_mask'].shape),'attention_mask_sum_total_nnz':int(micro_batch['attention_mask'].sum().item()),'scope':'ACTUAL_MICRO_INPUT_SHAPES_AND_MASK_SUM_NOT_SINGLE_SEQUENCE_PROOF_NOT_COMPLETED_UPDATE'}
 path=pathlib.Path(context['directory'])/('actor-pre-backward-memory-rank%d.jsonl'%context['rank'])
 raw=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 try:
  offset=0
  while offset<len(raw):
   count=os.write(fd,raw[offset:])
   if count<=0:raise OSError('Diagnostic append made no progress')
   offset+=count
  os.fsync(fd)
 finally:os.close(fd)
 return record
def observed_update(actor,original,data):
 import torch
 optimizer_steps=[];micro=[]
 before=sampled_parameters(actor)
 forward=actor._forward_micro_batch;step=actor._optimizer_step
 def observed_forward(*args,**kwargs):
  output=forward(*args,**kwargs);memory=pre_backward_memory(actor,torch,len(micro)+1,args[0] if args else kwargs.get('micro_batch'));micro.append(dict(epoch=time.time(),finite=all(bool(torch.isfinite(t).all()) for t in output),pre_backward_memory=memory));return output
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
  if len(updates)!=outer_steps or any(u['local_rows']!=320 for u in updates):raise ValueError('Actual2560/world8 rows missing')
  if any(u.get('actual_outer_update_ordinal')!=step or len(u['optimizer_step_records'])!=10 or len(u['micro_forward_records'])!=40 for step,u in enumerate(updates,1)):raise ValueError('Actual Search-R1 per-outer optimizer/micro sequence mismatch')
  if any(set(u['input_finite'])!={'old_log_probs','advantages','loss_mask','ref_log_prob'} or not all(u['input_finite'].values()) or any(not x['finite'] for x in u['micro_forward_records']) or any(not x['finite'] or not math.isfinite(x['grad_norm']) or x['grad_norm']<0 for x in u['optimizer_step_records']) for u in updates):raise ValueError('Nonfinite/negative actual update')
 return dict(scope='ACTUAL_SEARCH_R1_RESOURCE_UPDATES_UNREVIEWED_NOT_100_TRAINING',actual_outer_steps=outer_steps,optimizer_steps_per_rank=10*outer_steps,micro_forwards_per_rank=40*outer_steps,ranks=8)

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
   if r.get('cuda_allocator_environment')!=config['cuda_allocator']['value']:raise ValueError('Actual worker allocator environment differs')
   from allocator_policy import check_evidence
   check_evidence(r['cuda_allocator_evidence'])
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

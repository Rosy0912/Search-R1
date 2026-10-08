"""Wrap original methods in place: preserve canonical Worker base and constructor."""
import functools,os,pathlib,time
import torch
from resource_audit import atomic,observed_update
from allocator_policy import evidence as allocator_evidence

def pid_identity():
 stat=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=os.getpid(),starttime_ticks=int(stat[19]),cgroup=pathlib.Path('/proc/self/cgroup').read_text(),job=os.environ['SLURM_JOB_ID'],hostname=__import__('socket').gethostname())
def actual_gpu():
 from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetUUID,nvmlDeviceGetComputeRunningProcesses
 nvmlInit();pid=os.getpid();rows=[]
 for i in range(8):
  h=nvmlDeviceGetHandleByIndex(i);u=nvmlDeviceGetUUID(h);u=u.decode() if isinstance(u,bytes) else u
  for p in nvmlDeviceGetComputeRunningProcesses(h):
   if p.pid==pid:rows.append(u)
 return sorted(rows)

def install(worker_class):
 """No subclass or ctor patch. Caller serializes the owned module by value for Ray."""
 if worker_class.__dict__.get('_r1_observation_installed'):return worker_class
 original_init=worker_class.init_model;original_update=worker_class.update_actor
 @functools.wraps(original_init)
 def init_model(self,*args,**kwargs):
  output=original_init(self,*args,**kwargs);self._resource_updates=[]
  role='actor' if self._is_actor else 'ref';root=pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY']);root.mkdir(exist_ok=True,parents=True)
  record=dict(pid_identity(),role=role,rank=torch.distributed.get_rank(),gpu_uuids=actual_gpu(),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),ready_epoch=time.time(),cuda_allocator_environment=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),cuda_allocator_evidence=allocator_evidence(torch))
  atomic(root/(role+'-ready-rank%d.json'%record['rank']),record);return output
 @functools.wraps(original_update)
 def update_actor(self,data):
  if not self._is_actor:raise RuntimeError('Audit on wrong role')
  original=self.actor.update_policy
  def policy(data):
   self.actor._r1_forward_diagnostic_context=dict(pid_identity(),rank=torch.distributed.get_rank(),actual_outer_update_ordinal=len(self._resource_updates)+1,directory=os.environ['R1_RESOURCE_AUDIT_DIRECTORY']);result,receipt=observed_update(self.actor,original,data);receipt['actual_outer_update_ordinal']=len(self._resource_updates)+1;self._resource_updates.append(receipt);return result
  self.actor.update_policy=policy
  try:output=original_update(self,data)
  finally:self.actor.update_policy=original
  record=dict(pid_identity(),rank=torch.distributed.get_rank(),gpu_uuids=actual_gpu(),updates=self._resource_updates,actual_scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,actual_learning_rates_after_outer=[float(g['lr']) for g in self.actor_optimizer.param_groups])
  path=pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY'])/('actor-updates-rank%d-outer%d.json'%(record['rank'],len(self._resource_updates)))
  atomic(path,record);return output
 worker_class.init_model=init_model;worker_class.update_actor=update_actor;worker_class._r1_observation_installed=True
 return worker_class

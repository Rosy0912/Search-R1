"""Source-owned external observer around original registered Search-R1 FSDP methods."""
import json,os,pathlib,time
import torch
from verl.single_controller.base.decorator import register,Dispatch
from verl.workers.fsdp_workers import ActorRolloutRefWorker as OriginalWorker
from resource_audit import atomic,observed_update

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
class ObservedWorker(OriginalWorker):
 @register(dispatch_mode=Dispatch.ONE_TO_ALL)
 def init_model(self):
  output=super().init_model();self._resource_updates=[]
  role='actor' if self._is_actor else 'ref';root=pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY']);root.mkdir(exist_ok=True,parents=True)
  record=dict(pid_identity(),role=role,rank=torch.distributed.get_rank(),gpu_uuids=actual_gpu(),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),ready_epoch=time.time())
  atomic(root/(role+'-ready-rank%d.json'%record['rank']),record);return output
 @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
 def update_actor(self,data):
  if not self._is_actor:raise RuntimeError('Audit on wrong role')
  # Original update_actor loads original shards and calls actual actor.update_policy.
  original=self.actor.update_policy
  def policy(data):
   result,receipt=observed_update(self.actor,original,data);receipt['actual_outer_update_ordinal']=len(self._resource_updates)+1;self._resource_updates.append(receipt);return result
  self.actor.update_policy=policy
  try:output=super().update_actor(data)
  finally:self.actor.update_policy=original
  record=dict(pid_identity(),rank=torch.distributed.get_rank(),gpu_uuids=actual_gpu(),updates=self._resource_updates,actual_scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,actual_learning_rates_after_outer=[float(g['lr']) for g in self.actor_optimizer.param_groups])
  path=pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY'])/('actor-updates-rank%d-outer%d.json'%(record['rank'],len(self._resource_updates)))
  atomic(path,record);return output


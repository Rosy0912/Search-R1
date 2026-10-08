"""Observe foreign GPU use; cancel only this verified Slurm job on violation."""
import argparse,json,os,pathlib,subprocess,time,traceback
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
job=os.environ['SLURM_JOB_ID'];assert job.isdigit() and os.environ['SLURM_JOB_NODELIST']=='h100-8'
def has_job(text):
 return any('job_'+job in line.split(':',2)[-1].strip().split('/') for line in text.splitlines())
assert has_job(pathlib.Path('/proc/self/cgroup').read_text()), 'Guard must belong to this allocation cgroup'
expected=set(json.loads(pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/tools/expected-gpu-uuids.json').read_text()))
a.output.parent.mkdir(parents=True,exist_ok=True)
def append(record):
 with a.output.open('a') as out:out.write(json.dumps(record,allow_nan=False)+'\n')
from resource_ownership_v2 import OwnershipObserver
observer=OwnershipObserver(job, lambda record: append(dict(record,job_id=job)))
known_own=observer.known_own
zombie_proofs=observer.zombie_proofs
def own(pid):
 return observer.own(pid)
try:
 import pynvml as nv
 nv.nvmlInit();handles={}
 for i in range(nv.nvmlDeviceGetCount()):
  h=nv.nvmlDeviceGetHandleByIndex(i);u=nv.nvmlDeviceGetUUID(h)
  if isinstance(u,bytes):u=u.decode()
  if u in expected:handles[u]=h
 assert set(handles)==expected, 'Allocated UUID inventory unavailable'
 seen={u:int(time.time()*1e6) for u in handles}
 own_history={}
 while True:
  records=[];zombie_proofs.clear()
  for u,h in handles.items():
   mem=nv.nvmlDeviceGetMemoryInfo(h);util=nv.nvmlDeviceGetUtilizationRates(h)
   processes=nv.nvmlDeviceGetComputeRunningProcesses(h)+nv.nvmlDeviceGetGraphicsRunningProcesses(h)
   foreign_memory={};owners={}
   for proc in processes:
    belongs=own(proc.pid)
    if belongs is None:
     # PID exit race: it must have disappeared from the fresh GPU process list.
     assert proc.pid not in {x.pid for x in nv.nvmlDeviceGetComputeRunningProcesses(h)+nv.nvmlDeviceGetGraphicsRunningProcesses(h)}, 'Unknown active PID ownership'
     continue
    owners[proc.pid]=belongs
    if belongs:own_history[proc.pid]=time.time()*1e6
    if not belongs:
     assert isinstance(proc.usedGpuMemory,int) and 0<=proc.usedGpuMemory<=mem.total,'Unknown foreign GPU memory'
     foreign_memory[proc.pid]=max(foreign_memory.get(proc.pid,0),proc.usedGpuMemory)
   try:samples=nv.nvmlDeviceGetProcessUtilization(h,seen[u])
   except nv.NVMLError_NotFound:samples=[]  # NVML documents no sample entries.
   if samples:seen[u]=max(x.timeStamp for x in samples)
   foreign_sm={}
   for sample in samples:
    # Revalidate identity after the NVML sample read; do not trust an earlier
    # process-list owner snapshot across exit/PID reuse or a live cgroup move.
    belongs=own(sample.pid)
    if belongs is None and sample.pid in own_history and sample.timeStamp<=own_history[sample.pid]:belongs=True
    # A sampled departed PID may not be classifiable; stop if activity was high.
    if belongs is not True:
     assert 0<=sample.smUtil<=100, 'Unknown SM sample'
     foreign_sm[sample.pid]=max(foreign_sm.get(sample.pid,0),sample.smUtil)
   foreign=[{'pid':pid,'memory_bytes':memory} for pid,memory in foreign_memory.items()]
   memory_fraction=sum(foreign_memory.values())/mem.total
   # Conservative sum of per-process peaks over this polling interval.
   foreign_sm_upper=sum(foreign_sm.values())
   row=dict(uuid=u,total_memory_bytes=mem.total,total_used_memory_bytes=mem.used,total_gpu_util=util.gpu,foreign=foreign,foreign_memory_fraction=memory_fraction,foreign_sm_peaks=foreign_sm,foreign_sm_upper=foreign_sm_upper,nvml_samples=len(samples),identity_proven_own_zombies=[proof for pid,proof in zombie_proofs.items() if pid in set(owners)|{sample.pid for sample in samples}])
   records.append(row)
   assert memory_fraction<=.15 and foreign_sm_upper<=15, 'Foreign GPU activity exceeds user dual15% rule: '+json.dumps(row)
  append(dict(epoch=time.time(),job_id=job,gpus=records))
  a.output.with_suffix('.ready').touch()
  time.sleep(5)
except BaseException:
 append(dict(epoch=time.time(),job_id=job,event='STOP_OWN_JOB',error=traceback.format_exc()))
 subprocess.run(['scancel','--signal=TERM',job],check=False)
 raise

finally:
 observer.close()

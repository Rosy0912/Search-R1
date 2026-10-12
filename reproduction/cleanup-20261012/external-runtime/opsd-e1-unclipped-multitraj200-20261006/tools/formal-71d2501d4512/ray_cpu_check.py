"""Actual own fresh-Ray CPU worker check; disconnect only this driver."""
import json,os
from pathlib import Path
import ray
from owned_runtime import own,emit
root=Path(os.environ['E1_FORMAL_RUN']);job=os.environ['SLURM_JOB_ID']
ray.init(address=os.environ['RAY_ADDRESS'])
try:
 resources=ray.cluster_resources()
 if resources.get('GPU')!=8 or resources.get('CPU')!=80:raise ValueError('Wrong fresh Ray resource inventory')
 @ray.remote(num_cpus=1,num_gpus=0)
 def actual_worker():
  import os,re
  from pathlib import Path
  p=Path('/proc/self');status=(p/'status').read_text();cgroup=(p/'cgroup').read_text()
  return {'pid':os.getpid(),'job_id':os.environ.get('SLURM_JOB_ID'),'uid':os.getuid(),'uids':re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split(),'cgroup':cgroup,'start':(p/'stat').read_text().rsplit(')',1)[1].split()[19]}
 worker=ray.get(actual_worker.remote(),timeout=60);observed=own(worker['pid'])
 if worker['job_id']!=job or worker['uid']!=os.getuid() or set(map(int,worker['uids']))!={os.getuid()} or worker['start']!=observed['start'] or worker['cgroup']!=observed['cgroup']:raise ValueError('Foreign Ray worker')
 emit(root/'evidence/ray-actual-cpu-worker.json',{'resources':resources,'worker':worker,'driver':own(os.getpid()),'address':os.environ['RAY_ADDRESS'],'scope':'CPU task/current allocation only, not actor CUDA device or training proof'})
finally:ray.shutdown()

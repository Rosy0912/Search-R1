"""Read-only physical GPU snapshot using retained18 management authentication."""
import subprocess,json,datetime,os
host=os.uname().nodename
if host not in ('h100-5','h100-8','h100-10','h100-11'):raise RuntimeError('foreign host')
r={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'scope':'READONLY_PHYSICAL_GPU_NO_MUTATION'}
for k,args in [('gpus',['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);r[k]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r))

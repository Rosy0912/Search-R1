"""Read-only exact node/own job queue/GPU occupancy; never signals or submits."""
import subprocess,json,datetime,os
if os.uname().nodename!='h100-5':raise RuntimeError('wrong target')
v={'scope':'READONLY_CURRENT_NODE_JOB_GPU_NO_MUTATION','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for k,args in [('node',['scontrol','show','node','h100-5','-o']),('own3350',['scontrol','show','job','3350','-o']),('node_jobs',['squeue','-h','-w','h100-5','-o','%i|%j|%T|%R|%k']),('gpus',['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);v[k]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(v))

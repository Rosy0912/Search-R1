"""Explicit user requested cluster inventory; read-only scheduler metadata."""
import subprocess,json,datetime,os
if os.uname().nodename!='h100-5':raise RuntimeError('wrong management target')
r={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'READONLY_ALL_H_NODE_AND_EXPERIMENT_SCHEDULER_INVENTORY'}
for key,args in [('nodes',['scontrol','show','nodes','-o']),('jobs',['scontrol','show','jobs','-o']),('queue',['squeue','-h','-o','%i|%j|%T|%N|%R|%k']),('partitions',['sinfo','-N','-h','-o','%N|%T|%G|%C|%m'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);r[key]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r))

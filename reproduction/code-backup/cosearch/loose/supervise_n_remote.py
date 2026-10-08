import json,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');run=root/'runs/cosearch-local-gpu-five-20261003-n'
job=int(__import__('re').search(r'JobId=(\d+)', (run/'slurm-allocation.txt').read_text()).group(1))
record=dict(checked_at=time.time(),job=job,run=run.name)
queue=subprocess.run(['squeue','-h','-j',str(job),'-o','%i %T %N'],capture_output=True,text=True)
record['slurm']=queue.stdout.strip()
record['slurm_query_returncode']=queue.returncode
record['slurm_query_stderr']=queue.stderr.strip()
for key,name in [('probe','native-long-context-probe.json'),('driver_exit','driver-exit.json'),('acceptance','acceptance.json'),('warm_start','warm-start/manifest.json')]:
 if (run/name).exists():record[key]=json.loads((run/name).read_text())
if (run/'training.log').exists():
 capture=subprocess.run([str(root/'env/bin/python'),str(root/'tools/capture_worker_n.py')],capture_output=True,text=True)
 if capture.returncode==0:record['capture']=json.loads(capture.stdout.splitlines()[0])
 else:record['capture_pending_or_error']=capture.stderr[-1500:]
 audit=subprocess.run([str(root/'env/bin/python'),str(root/'tools/audit_gpu_n.py')],capture_output=True,text=True)
 record['audit']=audit.stdout[-1000:] if audit.returncode==0 else audit.stderr[-1000:]
from metric_log_source import validated_metric_log
text,record['metric_source']=validated_metric_log(run,root) if (run/'training.log').exists() else ('','training.log')
for key,tag in [('outer','COSEARCH_OUTER_METRICS_JSON='),('rank','COSEARCH_RANK_PROGRESS_JSON='),('progress','COSEARCH_BATCH_PROGRESS_JSON='),('first_batch','COSEARCH_FIRST_BATCH_JSON ')]:
 values=[]
 for line in text.splitlines():
  if tag in line:
   try:values.append(json.JSONDecoder().raw_decode(line.split(tag,1)[1])[0])
   except ValueError:pass
 record[key]=values if key=='outer' else values[-2:]
record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
for name in ('warm-start.log','retriever.log','native-ranker-probe.log','training.log','worker-training.log','reload.log'):
 if (run/name).exists():
  lines=(run/name).read_text(errors='replace').splitlines()
  record[name+'_tail']='\n'.join(lines[-4:])[-1800:]
reload=subprocess.run(['bash',str(root/'tools/start_five_reload_n.sh')],capture_output=True,text=True)
record['reload_advance']=dict(returncode=reload.returncode,stdout=reload.stdout[-1200:],stderr=reload.stderr[-1200:])
(run/'supervision-latest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))

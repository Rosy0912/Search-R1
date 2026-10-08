import json,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');run=root/'runs/cosearch-local-gpu-five-20261003-l'
record=dict(checked_at=time.time(),job=2988,run=run.name)
record['slurm']=subprocess.check_output(['squeue','-h','-j','2988','-o','%i %T %N'],text=True).strip()
for key,name in [('probe','native-long-context-probe.json'),('driver_exit','driver-exit.json'),('acceptance','acceptance.json')]:
 if (run/name).exists():record[key]=json.loads((run/name).read_text())
if (run/'training.log').exists():
 capture=subprocess.run([str(root/'env/bin/python'),str(root/'tools/capture_worker_l.py')],capture_output=True,text=True)
 if capture.returncode==0:record['capture']=json.loads(capture.stdout.splitlines()[0])
 else:record['capture_pending_or_error']=capture.stderr[-1500:]
 audit=subprocess.run([str(root/'env/bin/python'),str(root/'tools/audit_gpu_l.py')],capture_output=True,text=True)
 record['audit']=audit.stdout[-1000:] if audit.returncode==0 else audit.stderr[-1000:]
text=''
for name in ('training.log','worker-training.log'):
 if (run/name).exists():text+='\n'+(run/name).read_text(errors='replace')
for key,tag in [('outer','COSEARCH_OUTER_METRICS_JSON='),('rank','COSEARCH_RANK_PROGRESS_JSON='),('progress','COSEARCH_BATCH_PROGRESS_JSON=')]:
 values=[]
 for line in text.splitlines():
  if tag in line:
   try:values.append(json.JSONDecoder().raw_decode(line.split(tag,1)[1])[0])
   except ValueError:pass
 record[key]=values if key=='outer' else values[-2:]
record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
for name in ('retriever.log','native-ranker-probe.log','training.log','worker-training.log','reload.log'):
 if (run/name).exists():
  lines=(run/name).read_text(errors='replace').splitlines()
  record[name+'_tail']='\n'.join(lines[-4:])[-1800:]
reload=subprocess.run(['bash',str(root/'tools/start_five_reload_l.sh')],capture_output=True,text=True)
record['reload_advance']=dict(returncode=reload.returncode,stdout=reload.stdout[-1200:],stderr=reload.stderr[-1200:])
(run/'supervision-latest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))

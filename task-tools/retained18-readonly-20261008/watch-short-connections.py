"""Bounded read-only supervision. Fresh strict SSH per poll, durable evidence before wait."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
if sys.flags.optimize:raise RuntimeError("Optimized Python disables observation identity checks")
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
base=Path(__file__).resolve().parent
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
comment='BAPO_FRESH200:formal200:60f566b3ddabc88ce9de56ead5ad30a3d6795dd2:bab8981d9fd14afc98923eb4ebb89e33'
for i in range(31):
 started=time.monotonic();out=a.output_dir/('poll-%02d.json'%i)
 try:
  result=subprocess.run([sys.executable,str(base/'transport.py'),'--python-file',str(base/'read-slurm3315.py'),'--output',str(out)],capture_output=True,text=True,timeout=140)
 except subprocess.TimeoutExpired as exc:
  result=subprocess.CompletedProcess(exc.cmd,124,stdout='LOCAL_OBSERVER_TIMEOUT',stderr=str(exc))
 record={'poll':i,'observer_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transport_exit_code':result.returncode,'transport_stdout':result.stdout,'transport_stderr':result.stderr,'job_id':3315,'job_state':'UNKNOWN','training_terminal':False}
 if result.returncode==0:
  try:
   doc=json.loads(out.read_text());record['slurm']=doc
   assert isinstance(doc,dict) and type(doc.get('job_id')) is int and doc['job_id']==3315
   assert type(doc.get('exit_code')) is int and isinstance(doc.get('stdout'),str) and isinstance(doc.get('stderr'),str)
   if doc['exit_code']==0:
    lines=[line for line in doc['stdout'].splitlines() if line.strip()];assert len(lines)==1,'expected exactly one job record'
    pairs=[part.split('=',1) for part in lines[0].split() if '=' in part];keys=[pair[0] for pair in pairs];assert len(keys)==len(set(keys)),'duplicate Slurm field'
    fields=dict(pairs);assert fields.get('JobId')=='3315' and fields.get('Comment')==comment,'foreign job/source identity'
    state=fields.get('JobState');assert isinstance(state,str) and state,'missing job state'
    record.update(identity_ok=True,job_state=state,training_terminal=state in terminal,restarts=fields.get('Restarts'),requeue=fields.get('Requeue'))
  except (AssertionError,ValueError,OSError,TypeError,KeyError) as exc:
   record.update(job_state='UNKNOWN',training_terminal=False,parse_error=str(exc))
 with (a.output_dir/('observation-%02d.json'%i)).open('x') as f:
  f.write(json.dumps(record,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(record),flush=True)
 if record['training_terminal']:break
 if i<30:time.sleep(max(0,60-(time.monotonic()-started)))
print(json.dumps({'observer_window_complete':True,'training_terminal':record['training_terminal'],'last_job_state':record['job_state'],'job_id':3315}),flush=True)

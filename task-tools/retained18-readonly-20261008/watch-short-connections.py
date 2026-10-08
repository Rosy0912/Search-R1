"""Bounded read-only supervision. Fresh strict SSH per poll, durable evidence before wait."""
import argparse,datetime,json,re,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
base=Path(__file__).resolve().parent
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
for i in range(31):
 started=time.monotonic();out=a.output_dir/('poll-%02d.json'%i)
 try:
  result=subprocess.run([sys.executable,str(base/'transport.py'),'--python-file',str(base/'read-slurm3315.py'),'--output',str(out)],capture_output=True,text=True,timeout=140)
 except subprocess.TimeoutExpired as exc:
  result=subprocess.CompletedProcess(exc.cmd,124,stdout='LOCAL_OBSERVER_TIMEOUT',stderr=str(exc))
 record={'poll':i,'observer_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transport_exit_code':result.returncode,'transport_stdout':result.stdout,'transport_stderr':result.stderr,'job_id':3315,'job_state':'UNKNOWN','training_terminal':False}
 if result.returncode==0:
  doc=json.loads(out.read_text());record['slurm']=doc
  if doc.get('exit_code')==0:
   text=doc['stdout'];identity_ok='JobId=3315 ' in text and 'BAPO_FRESH200:formal200:60f566b3ddabc88ce9de56ead5ad30a3d6795dd2:bab8981d9fd14afc98923eb4ebb89e33' in text;record['identity_ok']=identity_ok
   state=re.search(r'\bJobState=(\S+)',text);record['job_state']=state.group(1) if state else 'UNKNOWN';record['training_terminal']=identity_ok and record['job_state'] in terminal
   if not identity_ok:record['job_state']='UNKNOWN_FOREIGN_IDENTITY'
 (a.output_dir/('observation-%02d.json'%i)).write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
 if record['training_terminal']:break
 if i<30:time.sleep(max(0,60-(time.monotonic()-started)))
print(json.dumps({'observer_window_complete':True,'training_terminal':record['training_terminal'],'last_job_state':record['job_state'],'job_id':3315}),flush=True)

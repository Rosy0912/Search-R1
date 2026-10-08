from pathlib import Path
import subprocess,json,time,re,datetime
W=Path(__file__).resolve().parents[1]
peek=W/'fresh200-deployment-r2/peek-3301.py'
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
for index in range(100):
 target=W/'evidence'/f'fresh200-r2-discovery128-3301-watch-{index:03d}.actual.json'
 err=target.with_suffix('.stderr')
 with target.open('x') as out,err.open('x') as error:
  result=subprocess.run(['python3',str(W/'h100_exec.py'),'--socks','--timeout','60','--stdin-file',str(peek),'python3 -B -'],stdout=out,stderr=error)
 event={'observation':index,'transport_exit':result.returncode,'path':str(target),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'READONLY_OBSERVATION_NO_SUBMIT_STOP_RESTART'}
 if result.returncode==0:
  try:
   doc=json.loads(target.read_text());m=re.search(r'JobState=(\S+)',doc['scontrol']['stdout']);state=m.group(1) if m else None
   event.update(job_state=state,native=doc.get('native_progress'),resource_reason=doc.get('latest_resource_watch',{}).get('reason'),pause=doc.get('latest_resource_watch',{}).get('pause'))
  except (ValueError,KeyError) as error:event['observation_error']=str(error);state=None
 else:state=None
 print(json.dumps(event),flush=True)
 if state in terminal:
  print('ACTUAL_TERMINAL_OBSERVED_NO_TRAINING_ACCEPTANCE',flush=True);break
 time.sleep(45)
else:print('OBSERVATION_TIMEOUT_NOT_TERMINAL_NO_RESTART',flush=True)

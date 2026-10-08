import subprocess,time,json,datetime
for i in range(61):
 p=subprocess.run(['scontrol','show','job','-o','3310'],capture_output=True,text=True)
 print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3310,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}),flush=True)
 if p.returncode==0 and any('JobState='+s in p.stdout for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL']):break
 time.sleep(30)

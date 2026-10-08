import subprocess,json,pathlib,time,os
job=3144;root='/dataset/shaoshitong/search-r1-rl/topd/logs/g1-initial3102-r4r10-allocation-evidence'
end=time.monotonic()+4*3600
while time.monotonic()<end:
 try:
  first=subprocess.run(['python3','work/g1_firstfault_snapshot.py',str(job),root],text=True,capture_output=True,timeout=180)
  live=subprocess.run(['python3','work/g1_live_snapshot.py',str(job),root],text=True,capture_output=True,timeout=180)
  if first.returncode or live.returncode:
   print(json.dumps({'scope':'READ_ONLY_TRANSPORT_FAILURE_NO_RETRY_MUTATION','first_returncode':first.returncode,'first_stderr':first.stderr[-1000:],'live_returncode':live.returncode,'live_stderr':live.stderr[-1000:]}),flush=True)
  else:
   f=json.loads(first.stdout);l=json.loads(live.stdout)
   names=[v['path'] for v in l['files']]
   print(json.dumps({'epoch':l['epoch'],'queue':l['queue'].strip(),'primary_errors':f['primary_error_artifact_count'],'firstfault_errors':f['errors'],'artifacts':names,'newest_tails':[{k:v for k,v in x.items() if k in ('path','tail')} for x in l['files'] if 'tail' in x]}),flush=True)
   if not l['queue'].strip():
    q=f['scontrol'];p=pathlib.Path('outputs/remote-evidence/g1-initial3144-actual-final-readonly.json')
    with p.open('x') as fd:json.dump({'epoch':f['epoch'],'job_id':job,'scontrol':q,'scope':'ACTUAL_READONLY_TERMINAL_NO_STATE_INFERENCE'},fd,indent=2);fd.flush();os.fsync(fd.fileno())
    break
 except Exception as e:print(json.dumps({'readonly_observer_error':repr(e)}),flush=True)
 time.sleep(30)

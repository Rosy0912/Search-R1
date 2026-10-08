import subprocess,pathlib,json,time,os
job=3176;root='/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v6/evidence'
body=r'''import pathlib,json,time,subprocess,hashlib
r=pathlib.Path(ROOT);out={'scope':'READ_ONLY_ACTUAL_STANDARD_SEARCH_R1_PROFILE_NOT_G1','job_id':3176,'epoch':time.time(),'files':[],'errors':[]}
for cmd,key in [(['scontrol','show','job','3176','-o'],'scontrol'),(['squeue','-h','-j','3176','-o','%i|%T|%M'],'queue')]:
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=15);out[key]={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
if r.exists():
 for p in sorted(r.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.suffix not in ('.json','.jsonl','.log'):continue
  try:
   st=p.stat();item={'relative_path':str(p.relative_to(r)),'size':st.st_size,'mtime_epoch':st.st_mtime}
   if p.suffix=='.json' and st.st_size<256*1024:
    b=p.read_bytes();item.update(sha256=hashlib.sha256(b).hexdigest(),data=json.loads(b))
   elif p.suffix in ('.jsonl','.log'):
    with p.open('rb') as f:f.seek(max(0,st.st_size-1500));item['tail']=f.read().decode(errors='replace')
   out['files'].append(item)
  except Exception as e:out['errors'].append({'path':str(p),'error':repr(e)})
print(json.dumps(out))
'''.replace('ROOT',repr(root))
out=pathlib.Path('outputs/remote-evidence/r1-standard32b-profile3176-readonly.jsonl');end=time.monotonic()+4*3600
while time.monotonic()<end:
 p=subprocess.run(['python3','work/remote_topd.py'],input="python3 - <<'OBSERVE'\n"+body+"\nOBSERVE\n",text=True,capture_output=True,timeout=60)
 if p.returncode:
  print(json.dumps({'transport_error':p.stderr[-1500:],'returncode':p.returncode}),flush=True)
 else:
  data=json.loads(p.stdout)
  with out.open('a') as f:f.write(json.dumps(data)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps({'epoch':data['epoch'],'queue':data['queue'],'files':[f['relative_path'] for f in data['files']],'errors':data['errors']}),flush=True)
  if any('JobState='+state in data['scontrol']['stdout'] for state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED')):
   final=pathlib.Path('outputs/remote-evidence/r1-standard32b-profile3176-actual-final.json')
   with final.open('x') as f:json.dump({k:v for k,v in data.items() if k!='files'},f,indent=2);f.flush();os.fsync(f.fileno())
   break
 time.sleep(45)

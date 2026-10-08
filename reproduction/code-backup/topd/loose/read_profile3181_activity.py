import subprocess,pathlib,json,hashlib
body=r'''import pathlib,json,hashlib,time,subprocess,os
P=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v7/evidence');records=[]
for q in sorted((P/'actor-resource-audit').glob('actor-ready-rank*.json')):
 x=json.loads(q.read_text());records.append(dict(x,ready_source_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
assert len(records)==8

def stat(b):
 s=b.decode();v=s[s.rfind(')')+2:].split();return {'state':v[0],'ppid':int(v[1]),'utime_ticks':int(v[11]),'stime_ticks':int(v[12]),'starttime_ticks':int(v[19])}
def capture():
 out={'epoch':time.time(),'workers':[],'errors':[]}
 for r in records:
  p=pathlib.Path('/proc')/str(r['pid'])
  try:
   b=(p/'stat').read_bytes();v=stat(b);cg=(p/'cgroup').read_bytes();status=(p/'status').read_text();uid=int(next(s for s in status.splitlines() if s.startswith('Uid:')).split()[1]);cmd=(p/'cmdline').read_bytes();wchan=(p/'wchan').read_text().strip()
   assert v['starttime_ticks']==r['starttime_ticks'];assert any('job_3181/step_batch' in x for x in cg.decode().splitlines());assert uid==0
   threads=[]
   for q in sorted((p/'task').iterdir()):
    try:tb=(q/'stat').read_bytes();tv=stat(tb);threads.append(dict(tid=int(q.name),**tv,comm=(q/'comm').read_text().strip()))
    except FileNotFoundError:pass
   out['workers'].append(dict(pid=r['pid'],rank=r['rank'],**v,uid=uid,cgroup=cg.decode().splitlines(),wchan=wchan,cmdline_role=cmd.split(b'\0')[0].decode(errors='replace') if cmd.startswith(b'ray::') else 'REDACTED_ARGUMENTS_COMMAND_BASENAME:'+pathlib.Path(cmd.split(b'\0')[0].decode(errors='replace')).name,raw_stat_sha256=hashlib.sha256(b).hexdigest(),raw_cgroup_sha256=hashlib.sha256(cg).hexdigest(),raw_cmdline_sha256=hashlib.sha256(cmd).hexdigest(),VmRSS=next((s for s in status.splitlines() if s.startswith('VmRSS:')),None),threads=threads,ready_source_sha256=r['ready_source_sha256']))
  except BaseException as e:out['errors'].append(dict(pid=r['pid'],error=repr(e)))
 q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);out['gpu_query']=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,sha256=hashlib.sha256(q.stdout.encode()).hexdigest());out['completed_epoch']=time.time()
 return out
first=capture();time.sleep(30);second=capture();print(json.dumps({'scope':'READ_ONLY_KNOWN8RANK_PROC_GPU_ACTIVITY_NOT_STACK_NOT_STAGE_IDENTIFICATION','job_id':3181,'first':first,'second':second,'first_exception_present':(P/'trainer-first-exception.json').exists(),'actor_updates_present':sorted(q.name for q in (P/'actor-resource-audit').glob('*updates*'))}))
'''
r=subprocess.run(['python3','work/remote_topd.py'],input="python3 - <<'ACTIVITY'\n"+body+"\nACTIVITY\n",text=True,capture_output=True,timeout=75)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);p=pathlib.Path('outputs/remote-evidence/standard3181-independent-30s-activity.json');p.write_text(json.dumps(x,indent=2)+'\n');print(p,hashlib.sha256(p.read_bytes()).hexdigest());print([(a['rank'],b['utime_ticks']+b['stime_ticks']-a['utime_ticks']-a['stime_ticks']) for a,b in zip(x['first']['workers'],x['second']['workers'])]);print(x['second']['gpu_query']['stdout']);print(x['first']['errors'],x['second']['errors'])

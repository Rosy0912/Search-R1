import datetime,json,os,subprocess
from pathlib import Path
pids=[3072946,3073263,3073264,3073265,3073266,3073267,3073268,3073269]
keys={'updates','skipped','mini_size','micro_size','offset','micro_offset','width'}
env=os.environ.copy();env['UV_CACHE_DIR']='/mnt/data/shaoshitong/search-r1-rl/opsd/cache/diag-uv'
cmd=['/mnt/data/shaoshitong/search-r1-rl/opsd/tools/uv-env/bin/uv','tool','run','--offline','--from','py-spy==0.4.1','py-spy','dump','--locals','--json','--pid']
result=[]
for pid in pids:
 r={'pid':pid,'sample_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 cg=Path(f'/proc/{pid}/cgroup')
 if not cg.exists():r['state']='exited';result.append(r);continue
 assert '/job_2875/' in cg.read_text(),'PID outside current owned job'
 call=subprocess.run(cmd+[str(pid)],capture_output=True,text=True,env=env,timeout=15)
 if call.returncode:r['state']='sample_failed';result.append(r);continue
 def walk(x):
  if isinstance(x,dict):
   if x.get('name')=='update_policy' and x.get('filename','').endswith('/search_r1/baselines/actor.py'):
    r['state']='update_policy';r['line']=x.get('line')
    for v in x.get('locals',[]):
     if v.get('name') in keys:
      raw=v.get('repr','')
      if raw.isdigit():r[v['name']]=int(raw)
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(json.loads(call.stdout));result.append(r)
print(json.dumps({'job_id':'2875','samples':result,'scope':'Read-only asynchronous frame integer samples; not completed-outer acceptance'},indent=2))

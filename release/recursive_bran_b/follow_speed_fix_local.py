"""One-shot local small-receipt followup; no App automation or training mutations."""
import argparse,json,pathlib,re,shlex,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--binding',required=True);p.add_argument('--until-step',type=int,default=10);a=p.parse_args();d=json.loads(pathlib.Path(a.binding).read_bytes());out=pathlib.Path(a.output)
cmd=shlex.join([d['view'],'env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',d['python'],'-B',d['reader'],'--ray-address',d['ray_address'],'--pids',','.join(map(str,d['worker_pids'])),'--run-id',d['run_id'],'--source-commit',d['source_commit']])
script='''from pathlib import Path
import json,time,re
root=Path(%r);main=next(x for x in root.glob('worker-*.out') if x.name.endswith('-%d.out'));text=main.read_text(errors='replace');metrics={}
for line in text.splitlines():
 m=re.search(r'step:(\d+) -',line)
 if m:metrics[m.group(1)]={k:float(v) for k,v in re.findall(r'([a-zA-Z0-9_/-]+):(-?[0-9.eE+-]+)',line)}
errors=[]
for x in root.glob('worker-*-%d.err'):errors.extend(v[:500] for v in x.read_text(errors='replace').splitlines() if 'Traceback' in v or 'PolicySafetyRefusal:' in v or 'RuntimeError:' in v)
print(json.dumps(dict(time=time.time(),metrics=metrics,errorlines=errors,next_steps=[int(x) for x in re.findall(r'epoch \d+, step (\d+)',text)],main_bytes=main.stat().st_size)))
'''%(d['logs'],d['main_pid'],d['main_pid'])
start=time.monotonic()
while time.monotonic()-start<21600:
 now=int(time.time());cache=subprocess.run(['ssh','hkust5',cmd],capture_output=True);(out/('speed-fix-follow-%d-cache.json'%now)).write_bytes(cache.stdout);(out/('speed-fix-follow-%d-cache.stderr'%now)).write_bytes(cache.stderr)
 stage=subprocess.run(['ssh','hkust5','python3 -'],input=script.encode(),capture_output=True);(out/('speed-fix-follow-%d-stage.json'%now)).write_bytes(stage.stdout);(out/('speed-fix-follow-%d-stage.stderr'%now)).write_bytes(stage.stderr)
 if cache.returncode or stage.returncode:print(json.dumps(dict(time=time.time(),observer_rc=[cache.returncode,stage.returncode],training_rc_unknown=True)),flush=True);break
 c=json.loads(cache.stdout);s=json.loads(stage.stdout);w=c['workers'];print(json.dumps(dict(time=time.time(),calls=[x['actual_optimizer_calls'] for x in w],scheduler=[x['scheduler'] for x in w],gen=[x['cache_generation_calls'] for x in w],sync=[x['total_weight_sync_calls'] for x in w],completed_steps=sorted(map(int,s['metrics'])),latest_step=max(s['next_steps'],default=0),errors=s['errorlines'])),flush=True)
 if s['errorlines']:break
 if all(str(i) in s['metrics'] for i in range(4,a.until_step+1)) and max(s['next_steps'],default=0)>a.until_step and all(x['scheduler']>=a.until_step for x in w):
  (out/'speed-fix-target-complete-and-next.json').write_text(json.dumps(dict(time=time.time(),cache=c,stage=s),indent=2));break
 time.sleep(60)

import pathlib,subprocess,json,time,re,shlex
p=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/recursive-bran-experiment-b-20261011')
T='/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011'
S='/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011'
command=shlex.join([T+'/source-speed-17fa8c1/release/recursive_bran_b/view_hkust5.sh','env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/shaoshitong/search-e1-config-study-20261004/env/bin/python','-B',S+'/source-monitor-06a0843/release/recursive_bran_b/read_speed_progress.py','--ray-address','10.123.4.12:61007','--pids','519768,520068,520069,520070'])
logscript='''from pathlib import Path
import json,time,re
root=Path('/home/shaoshitong/rbb5-speed-ray/ray/session_latest/logs');main=next(p for p in root.glob('worker-*.out') if p.name.endswith('-518921.out'));lines=main.read_text(errors='replace').splitlines();done=[l for l in lines if 'step:4 -' in l];metrics={}
if done:metrics={k:float(v) for k,v in re.findall(r'([a-zA-Z0-9_/-]+):(-?[0-9.eE+-]+)',done[-1])}
errfiles=[p for p in root.glob('worker-*.err') if p.name.endswith('-518921.err')];errorlines=[]
for p in errfiles:errorlines.extend(l[:500] for l in p.read_text(errors='replace').splitlines() if 'Traceback' in l or 'ValueError:' in l or 'RuntimeError:' in l)
print(json.dumps(dict(time=time.time(),step4_complete=bool(done),step5_seen=any('epoch 0, step 5' in l for l in lines),metrics=metrics,errorlines=errorlines,main_log_bytes=main.stat().st_size)))
'''
start=time.monotonic();i=0
while time.monotonic()-start<7200:
 i+=1;now=int(time.time());a=subprocess.run(['ssh','hkust5',command],capture_output=True);(p/('speed-follow-%d-cache.json'%now)).write_bytes(a.stdout);(p/('speed-follow-%d-cache.stderr'%now)).write_bytes(a.stderr)
 b=subprocess.run(['ssh','hkust5','python3 -'],input=logscript.encode(),capture_output=True);(p/('speed-follow-%d-stage.json'%now)).write_bytes(b.stdout);(p/('speed-follow-%d-stage.stderr'%now)).write_bytes(b.stderr)
 if a.returncode or b.returncode:print(json.dumps(dict(time=time.time(),status='OBSERVATION_CALL_FAILED_NOT_TRAINING_RC',a_rc=a.returncode,b_rc=b.returncode)),flush=True);break
 x=json.loads(a.stdout);y=json.loads(b.stdout);print(json.dumps(dict(time=time.time(),calls=[r['actual_optimizer_calls'] for r in x['workers']],scheduler=[r['scheduler'] for r in x['workers']],gen=[r['cache_generation_calls'] for r in x['workers']],sync=[r['total_weight_sync_calls'] for r in x['workers']],step4_complete=y['step4_complete'],step5_seen=y['step5_seen'],errorlines=y['errorlines'])),flush=True)
 if y['errorlines']:break
 if y['step4_complete'] and y['step5_seen'] and all(r['actual_optimizer_calls']>=80 and r['scheduler']>=4 for r in x['workers']):
  (p/'speed-first-complete-outer-and-next.json').write_text(json.dumps(dict(time=time.time(),cache=x,stage=y),indent=2));break
 time.sleep(60)

/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,datetime,hashlib\nbase=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011')\nrun=json.loads((base/'logs/run-manifest.json').read_text());assert run['run_id']=='15bb15365346401da1d919e9cde8f813'\no={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':run['run_id'],'source_commit':run['source_commit'] if 'source_commit' in run else None,'remote_base':str(base),'raw_small_receipts':{}}\nfor name in ['five-updates-acceptance.json','formal-inline-development-after.json']:\n p=base/'logs'/name\n if p.exists():\n  raw=p.read_bytes();o['raw_small_receipts'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'document':json.loads(raw)}\ncp=base/'checkpoints/search-g1-common8/actor/global_step_5'\no['checkpoint_files']={p.relative_to(cp).as_posix():p.stat().st_size for p in cp.rglob('*') if p.is_file()} if cp.exists() else {}\nfor name in ['complete.json','training_state.json','training_state_manifest.json']:\n p=cp/name\n if p.exists():\n  raw=p.read_bytes();o['raw_small_receipts']['checkpoint/'+name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'document':json.loads(raw)}\no['all_checkpoint_files_independently_read_back']=False\no['actual_updates']=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]\no['candidate_pools']=[{'attempt':p.name,'files':{f.name:f.stat().st_size for f in p.iterdir() if f.is_file()}} for p in (base/'artifacts/mixed-candidate-pools').iterdir() if p.is_dir()]\nprint(json.dumps(o))\n"
worker_cmd='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(observer)
mgmt=r'''import json,shlex,subprocess,datetime
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER_CMD],capture_output=True,text=True,timeout=65)
print(json.dumps({'route':'retained18 SSH-W h100-8','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''.replace('WORKER_CMD',repr(worker_cmd))
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host']+['env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)],capture_output=True,text=True,timeout=85)
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3364-ACTUAL-FIVE-CLOSURE-20261011-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY

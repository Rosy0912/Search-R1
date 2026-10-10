/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,subprocess,datetime,hashlib\nbase=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-a76d1a5-20261011')\nout={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}\nr=subprocess.run(['scontrol','show','job','3361','-o'],capture_output=True,text=True);out['job']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}\nfor name in ['pipeline-status.json','process-ledger.json','retrieval-staging-status.json','cleanup-status.json']:\n p=base/'logs'/name\n if p.is_file():\n  d=json.loads(p.read_text())\n  if name=='process-ledger.json':d={'jobs':{k:{'pid':v['leader']['pid'],'state':v['state'],'exit_code':v.get('exit_code')} for k,v in d['jobs'].items()}}\n  out[name]=d\nout['logs']={}\nfor p in (base/'logs').glob('*.log'):\n with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')\n out['logs'][p.name]={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'tail':tail}\nclaim=base/'logs/retrieval-staging-claim.json'\nif claim.exists():\n root=pathlib.Path(json.loads(claim.read_text())['root']);out['staging_files']={p.name:p.stat().st_size for p in root.iterdir() if p.is_file()}\nfor folder in ['artifacts/mixed-selection','logs/model-smoke-initial','logs/model-smoke-coexistence']:\n p=base/folder;out[folder]=[x.name for x in p.glob('*')] if p.is_dir() else []\nprint(json.dumps(out))\n"
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
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3361-ACTUAL-20261011-v2.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY

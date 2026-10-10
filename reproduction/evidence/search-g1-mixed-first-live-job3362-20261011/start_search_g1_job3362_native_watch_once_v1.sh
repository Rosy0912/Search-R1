/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,subprocess,datetime,os,hashlib\nbase=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-bb1fc86-20261011')\nsource=base/'repo/reproduction/evidence/search-g1-mixed-first-live-job3361-20261011/native_terminal_watch.py'\nassert hashlib.sha256(source.read_bytes()).hexdigest()=='3203bc8d8dde7288666e9ff57ff6804da59fc458577de37ce491b9c4fc6cd88b'\nintent=base/'preparation/native-watch-intent.json';receipt=base/'preparation/native-watch-receipt.json'\nassert not intent.exists() and not receipt.exists()\nwith intent.open('x') as f:json.dump({'job':3362,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':'3203bc8d8dde7288666e9ff57ff6804da59fc458577de37ce491b9c4fc6cd88b'},f);f.flush();os.fsync(f.fileno())\nlog=(base/'logs/native-watch.log').open('xb')\np=subprocess.Popen(['/usr/bin/python3','-B',str(source),'--base',str(base),'--job','3362'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nvalue={'pid':p.pid,'job':3362,'source_sha256':'3203bc8d8dde7288666e9ff57ff6804da59fc458577de37ce491b9c4fc6cd88b','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}\nwith receipt.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())\nprint(json.dumps(value))\n"
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
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3362-NATIVE-WATCH-20261011-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':False},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY

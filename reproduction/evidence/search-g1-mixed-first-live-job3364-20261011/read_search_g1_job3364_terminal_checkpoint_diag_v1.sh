/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,zipfile,pickletools,datetime,hashlib,subprocess,os\nbase=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011');cp=base/'checkpoints/search-g1-common8/actor/global_step_5';state=cp/'training_state'\no={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':json.loads((state/'manifest.json').read_text()),'pickle_globals':{}}\nfor p in state.glob('*.pt'):\n with zipfile.ZipFile(p) as z:\n  name=next(n for n in z.namelist() if n.endswith('/data.pkl'));raw=z.read(name)\n  o['pickle_globals'][p.name]={'pickle_bytes':len(raw),'globals':sorted(set(str(v) for op,v,_ in pickletools.genops(raw) if op.name=='GLOBAL'))}\nlib=base/'env/lib/python3.10/site-packages/torch'\no['torch_version_source']=(lib/'version.py').read_text()\nlines=(lib/'serialization.py').read_text().splitlines();i=next(i for i,v in enumerate(lines) if 'del torch._utils._thread_local_state.map_location' in v);o['serialization_fragment']='\\n'.join(lines[i-8:i+6])\no['owned_pid_ps']=subprocess.run(['ps','-p','3331500,3343201,3344087,3344503,3344504,3344505,3344506,3344507,3344508,3344509','-o','pid,ppid,lstart,stat,args'],capture_output=True,text=True).stdout\nr=subprocess.run(['scontrol','show','job','3364','-o'],capture_output=True,text=True);o['job']=r.stdout\no['terminal_log_sha']={}\nassert 'JobState=FAILED' in r.stdout\nfor p in (base/'logs').glob('*.log'):\n before=p.stat();h=hashlib.sha256()\n with p.open('rb') as f:\n  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)\n after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)\n o['terminal_log_sha'][p.name]={'inode':after.st_ino,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'full_read_sha256':h.hexdigest()}\nprint(json.dumps(o))\n"
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
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3364-ACTUAL-TERMINAL-CHECKPOINT-DIAG-20261011-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY

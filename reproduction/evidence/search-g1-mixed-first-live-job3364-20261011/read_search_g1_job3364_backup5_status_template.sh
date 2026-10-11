/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,os,socket,datetime,hashlib\nassert socket.gethostname()=='h100-10'\nroot=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-first-independent-backup-20261011/job3364-15bb15365346-step5-once-v1')\no={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'namespace':str(root),'namespace_exists':root.is_dir(),'read_only':True}\nif root.is_dir():\n assert root.resolve()==root\n for name in ['INTENT.json','RECEIPT.json','PROGRESS.json','RESULT.json']:\n  p=root/name\n  if p.exists():o[name]=json.loads(p.read_text())\n p=root/'independent_backup.py'\n if p.exists():o['actual_code_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()\n receipt=o.get('RECEIPT.json')\n if receipt:\n  proc=pathlib.Path('/proc')/str(receipt['pid']);st=proc/'stat'\n  if st.exists():\n   tail=st.read_text().rsplit(')',1)[1].split();ticks=int(tail[19]);argv=(proc/'cmdline').read_bytes().replace(b'\\x00',b' ').decode(errors='replace')\n   o['process']={'pid':receipt['pid'],'state':tail[0],'start_ticks':ticks,'sid':os.getsid(receipt['pid']),'same_start_ticks':ticks==receipt['start_ticks'],'argv':argv}\n  else:o['process']={'pid':receipt['pid'],'gone':True}\n for name in ['checkpoint.partial','checkpoint']:\n  p=root/name\n  if p.is_dir():o[name]={'exists':True,'files':{str(f.relative_to(p)):f.stat().st_size for f in p.rglob('*') if f.is_file()}}\n p=root/'backup.log'\n if p.exists():\n  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));o['log_tail']=f.read().decode(errors='replace')\nprint(json.dumps(o))\n"
worker_cmd='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(observer)
mgmt=r'''import json,shlex,subprocess,datetime
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-10';ip=cfg['gateway_ip'];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER_CMD],capture_output=True,text=True,timeout=65)
print(json.dumps({'route':'retained18 h100-10 independent backup','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''.replace('WORKER_CMD',repr(worker_cmd))
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host']+['env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)],capture_output=True,text=True,timeout=85)
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3364-INDEPENDENT-BACKUP5-STATUS-20261011-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY

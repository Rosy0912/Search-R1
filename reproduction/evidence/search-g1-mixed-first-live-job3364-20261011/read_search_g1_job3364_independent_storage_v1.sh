/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,os,subprocess,socket,datetime\nroot=pathlib.Path('/mnt/shared/workspace/training_data_node_8/shaoshitong/search-g1-mixed-2de313e-20261011')\nfs=os.statvfs('/mnt/data')\no={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'ordinary_free_bytes':fs.f_bavail*fs.f_frsize,'physical_free_bytes':fs.f_bfree*fs.f_frsize,'destination_device':os.stat('/mnt/data').st_dev,'source_base_exists':root.is_dir(),'destination_mount':subprocess.check_output(['findmnt','-n','-T','/mnt/data','-o','SOURCE,FSTYPE'],text=True).strip()}\nif root.is_dir():\n o['source_device']=root.stat().st_dev\n o['source_mount']=subprocess.check_output(['findmnt','-n','-T',str(root),'-o','SOURCE,FSTYPE'],text=True).strip()\n p=root/'logs/run-manifest.json'\n if p.exists():\n  d=json.loads(p.read_text());o['source_run']={k:d.get(k) for k in ['run_id','source_commit','checkpoint_dir','profile','mixed_first_selection']}\nprint(json.dumps(o))\n"
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
print(json.dumps({'route':'retained18 SSH-W h100-8','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''.replace('WORKER_CMD',repr(worker_cmd))
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host']+['env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)],capture_output=True,text=True,timeout=85)
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3364-INDEPENDENT-STORAGE-20261011-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY

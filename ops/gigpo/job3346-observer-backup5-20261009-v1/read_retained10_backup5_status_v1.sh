/usr/bin/python3 - <<'PY'
import ast,sys,subprocess,shlex,json,datetime
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
module=ast.parse(Path('work/read_gigpo_job3332_actual.py').read_text())
observer="import json,pathlib,subprocess,datetime,os\nroot=pathlib.Path('/mnt/data/shaoshitong/gigpo-e1-instruct200-independent-backup-20261009/backup5-once-v1')\nout={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':subprocess.check_output(['hostname'],text=True).strip(),'records':{}}\nfor n in ['INTENT.json','RECEIPT.json','PROGRESS.json','RESULT.json']:\n p=root/n\n if p.exists():out['records'][n]=json.loads(p.read_text())\nr=out['records'].get('RECEIPT.json',{});pid=r.get('pid')\nif pid:\n p=subprocess.run(['ps','-p',str(pid),'-o','pid,stat,args'],capture_output=True,text=True);out['process']={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}\nout['partial_files']=[{'relative_path':str(p.relative_to(root)),'bytes':p.stat().st_size} for p in (root/'checkpoint5.partial').rglob('*') if p.is_file()]\nout['final_exists']=(root/'checkpoint5').exists();out['backup_log']=(root/'backup.log').read_text(errors='replace')[-2000:]\nprint(json.dumps(out))\n"
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
p=Path('outputs/gigpo-e1-instruct200/RETAINED10-BACKUP5-ACTUAL-STATUS-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
o=json.loads(m['stdout']);print(json.dumps(o));raise SystemExit(0);print(json.dumps({'utc':o['utc'],'slurm':o['slurm'],'gpu':o['gpu'],'runs':o['runs'],'slurm_tail':o['slurm_log'][-2500:] if o['slurm_log'] else None},ensure_ascii=False))
PY

"""Task-scoped strict retained18 SSH transport. No credentials copied or globals changed."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006'
p=argparse.ArgumentParser();p.add_argument('--target',choices=['h100-5','h100-8','h100-10','h100-11'],default='h100-5');p.add_argument('--python-file',required=True);p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=100);a=p.parse_args();assert 1<=a.timeout<=1950
src=Path(a.python_file).read_text()
command='cd '+shlex.quote(ROOT+'/repo')+' && CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(ROOT+'/repo')+' '+shlex.quote(ROOT+'/env/bin/python')+' -B -c '+shlex.quote(src)
if a.target!='h100-5':command='env -u LD_LIBRARY_PATH python3 -B -c '+shlex.quote(src)
code='''import json,shlex,subprocess,datetime
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target=TARGET;ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
'''.replace('target=TARGET;', 'target='+repr(a.target)+';')+ 'TIMEOUT='+repr(a.timeout)+'\ncommand='+repr(command)+'''\np=subprocess.run(args+[user+'@'+ip,command],capture_output=True,text=True,timeout=TIMEOUT)
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}),flush=True)
'''
args=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=15','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host','env -u LD_LIBRARY_PATH python3 -']
r=subprocess.run(args,input=code,capture_output=True,text=True,timeout=a.timeout+25)
base=Path(a.output);base.parent.mkdir(parents=True,exist_ok=True)
Path(str(base)+'.transport.stderr').write_text(r.stderr)
Path(str(base)+'.transport.stdout').write_text(r.stdout)
if r.returncode:sys.exit(r.returncode)
d=json.loads(r.stdout);Path(str(base)+'.record.json').write_text(json.dumps(d,indent=2));base.write_text(d['stdout']);print(json.dumps({'output':str(base),'observed_utc':d['observed_utc'],'exit_code':d['exit_code'],'stderr':d['stderr']}));sys.exit(d['exit_code'])

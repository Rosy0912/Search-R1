import subprocess,shlex,json,datetime
from pathlib import Path
MGMT="import json,shlex,subprocess,datetime,sys\nfrom pathlib import Path\ncfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())\ntarget='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']\nprefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']\nargs=prefix.copy()\nif ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]\nkeys=cfg.get('trusted_target_host_keys',{}).get(target)\nif keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]\nr=subprocess.run(args+[user+'@'+ip,WORKER_CMD],input=sys.stdin.buffer.read(),capture_output=True,timeout=65)\nprint(json.dumps({'route':'retained18 SSH-W h100-8','returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))\n"
def remote(command,evidence,stdin=None,emit=True):
    mgmt=MGMT.replace('WORKER_CMD',repr(command))
    args=['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host','env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)]
    r=subprocess.run(args,input=stdin,capture_output=True,timeout=85)
    o=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode())
    with Path(evidence).open('x') as f:json.dump(o,f,indent=2)
    if r.returncode:raise RuntimeError(o['stderr'])
    m=json.loads(o['stdout']);assert m['returncode']==0,m
    if emit:print(m['stdout'])
    return m['stdout']

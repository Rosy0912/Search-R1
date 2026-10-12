"""Stream only a small Git code bundle via the target session's existing route."""
import argparse,base64,hashlib,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--commit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
blob=pathlib.Path(a.bundle).read_bytes();sha=hashlib.sha256(blob).hexdigest()
remote="""import sys,pathlib,subprocess,hashlib,json
b=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012');f=b/'control/code.bundle';f.write_bytes(sys.stdin.buffer.read());assert hashlib.sha256(f.read_bytes()).hexdigest()==SHA
assert not (b/'source').exists()
subprocess.run(['git','clone',str(f),str(b/'source')],check=True)
subprocess.run(['git','-C',str(b/'source'),'checkout','--detach',COMMIT],check=True)
assert subprocess.check_output(['git','-C',str(b/'source'),'rev-parse','HEAD'],text=True).strip()==COMMIT
print(json.dumps({'code_deployed':True,'commit':COMMIT,'bundle_bytes':f.stat().st_size,'bundle_sha256':SHA}))
""".replace('SHA',repr(sha)).replace('COMMIT',repr(a.commit))
bridge="""import json,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text());target='h100-11';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3'];args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,COMMAND],input=sys.stdin.buffer.read(),capture_output=True,timeout=180);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
""".replace('COMMAND',repr('env -u LD_LIBRARY_PATH python3 -c '+shlex.quote(remote)))
args=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=15','-oProxyCommand=none','-oProxyJump=none','mgmt-host','env -u LD_LIBRARY_PATH python3 -c '+shlex.quote(bridge)]
r=subprocess.run(args,input=blob,capture_output=True,timeout=220)
pathlib.Path(a.output).write_bytes(r.stdout);pathlib.Path(a.output+'.stderr').write_bytes(r.stderr)
print(json.dumps({'rc':r.returncode,'code_bytes':len(blob),'stdout':r.stdout.decode(),'stderr':r.stderr.decode()}));raise SystemExit(r.returncode)

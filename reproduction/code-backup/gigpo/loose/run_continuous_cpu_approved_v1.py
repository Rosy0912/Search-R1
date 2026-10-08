"""Invoke only an original independent report's exact, once-only CPU policy."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();role=sys.argv[1];assert role in ('source','runtime')
name='SOURCE11' if role=='source' else 'RUNTIME12';local=base/('outputs/gigpo-e1-instruct200/CONTINUOUS-PREEXEC-'+name+'-REVIEW-v2.json');raw=local.read_bytes();review=json.loads(raw);policy=review['approved_cpu_policy'];sha=hashlib.sha256(raw).hexdigest()
assert policy['role']==role and policy['single_execution_only'] is True and policy['GPU_allowed'] is False and review['blocking_findings']==[]
root='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
assert policy['management_helper']==dict(path=root+'/tools/management/continuous-cpu-once-v4.py',bytes=19553,sha256='520819b9e7954ecb66b45c17a673af40497b0e4a8f6ee045efb78352f84ad46f')
snapshot=root+'/logs/'+('FRESH200-SOURCE-STATIC-SNAPSHOT-v11.json' if role=='source' else 'FRESH200-CONTINUATION-STATIC-MANIFEST-v12.json')
payload=dict(review_path=policy['original_review_path'],review_sha=sha,review_b64=base64.b64encode(raw).decode(),helper=policy['management_helper'],snapshot_path=snapshot,snapshot_sha=policy['snapshot_sha256'],output=policy['output_directory'])
code='''import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);path=Path(p['review_path']);assert path.is_relative_to('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/logs') and path.parent.resolve()==path.parent
raw=base64.b64decode(p['review_b64']);assert hashlib.sha256(raw).hexdigest()==p['review_sha']
if path.exists():assert path.read_bytes()==raw
else:
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
helper=Path(p['helper']['path']);h=helper.read_bytes();assert len(h)==p['helper']['bytes'] and hashlib.sha256(h).hexdigest()==p['helper']['sha256']
assert not Path(p['output']).exists(),'Existing CPU claim: inspect original output, never repeat'
cmd=['/usr/bin/python3',str(helper),str(path),p['review_sha'],p['snapshot_path'],p['snapshot_sha'],p['output']]
r=subprocess.run(cmd,text=True,capture_output=True)
print(json.dumps(dict(command=cmd,management_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,CPU_only=True,GPU_executed=False,formal_iterations=0),indent=2))
raise SystemExit(r.returncode)
'''
inner='/usr/bin/python3 -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner);route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
r=subprocess.run(route,input=json.dumps(payload).encode(),capture_output=True,timeout=600)
out=base/('outputs/gigpo-e1-instruct200/CPU-'+name+'-ACTUAL-INVOKE-v1.json')
with out.open('xb') as f:f.write(r.stdout if r.stdout else json.dumps(dict(route_returncode=r.returncode,route_stderr=r.stderr.decode(errors='replace'))).encode());f.write(b'\n')
if r.returncode:raise RuntimeError('CPU invocation failed; preserved '+str(out)+' '+r.stderr.decode(errors='replace'))
x=json.loads(r.stdout);print(json.dumps(dict(role=role,management_returncode=x['management_returncode'],summary=x['stdout'],stderr=x['stderr'],invocation_record=str(out),formal_iterations=0)))

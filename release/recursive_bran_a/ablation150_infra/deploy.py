"""Deploy a published Git code archive through the retained management route.

Only code/configuration are transferred. Original runs and retriever stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--management-socket', required=True)
p.add_argument('--archive', required=True)
p.add_argument('--commit', required=True)
p.add_argument('--manifest', required=True)
p.add_argument('--destination', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
blob = Path(a.archive).read_bytes()
manifest = json.loads(Path(a.manifest).read_text())
assert manifest['commit'] == a.commit
assert len(a.commit) == 40
assert a.destination.startswith('/dataset/shaoshitong/recursive-ablation150-')
digest = hashlib.sha256(blob).hexdigest()
remote = '''import hashlib,io,json,pathlib,sys,tarfile,time
b=pathlib.Path(DEST);manifest=json.loads(sys.stdin.buffer.readline())
if (b/'deployment.json').exists():
 old=json.loads((b/'deployment.json').read_text())
 assert old['commit']==COMMIT
 for name,sha in manifest.items():assert hashlib.sha256((b/'source'/name).read_bytes()).hexdigest()==sha
 print(json.dumps(old));raise SystemExit(0)
assert not (b/'source').exists(), 'Unreceipted deployment exists; inspect it'
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==DIGEST
b.mkdir(parents=True,exist_ok=True);source=b/'source';source.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for member in tar.getmembers():
  target=pathlib.PurePosixPath(member.name)
  assert not target.is_absolute() and '..' not in target.parts and (member.isfile() or member.isdir())
 for member in tar.getmembers():
  target=source/member.name
  if member.isdir():target.mkdir(parents=True,exist_ok=True)
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(tar.extractfile(member).read());target.chmod(member.mode & 0o777)
for name,sha in manifest.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
record=dict(time=time.time(),commit=COMMIT,source=str(source),archive_bytes=len(raw),archive_sha256=DIGEST,
 file_count=len(manifest),all_manifest_sha_equal=True,original_runs_modified=False)
(b/'deployment.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
'''
remote = remote.replace('DEST', repr(a.destination)).replace('COMMIT', repr(a.commit)).replace('DIGEST', repr(digest))
bridge = '''import json,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-11';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,COMMAND],input=sys.stdin.buffer.read(),capture_output=True,timeout=180)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
'''.replace('COMMAND', repr('env -u LD_LIBRARY_PATH python3 -c ' + shlex.quote(remote)))
args = ['ssh', '-S', a.management_socket, '-oControlMaster=no', '-T', '-oBatchMode=yes', '-oConnectTimeout=15', '-oProxyCommand=none', '-oProxyJump=none', 'mgmt-host', 'env -u LD_LIBRARY_PATH python3 -c ' + shlex.quote(bridge)]
r = subprocess.run(args, input=json.dumps(manifest['files']).encode() + b'\n' + blob, capture_output=True, timeout=220)
Path(a.output).write_bytes(r.stdout)
Path(a.output + '.stderr').write_bytes(r.stderr)
print(json.dumps(dict(rc=r.returncode, archive_bytes=len(blob), stdout=r.stdout.decode(), stderr=r.stderr.decode())), flush=True)
raise SystemExit(r.returncode)

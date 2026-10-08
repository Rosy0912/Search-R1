"""Upload only an immutable source/metadata package to h100-9; no GPU launch."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from remote import route
payload=json.loads(Path(sys.argv[1]).read_bytes())
remote=r'''
import base64,hashlib,json,os,socket,sys,tarfile
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
p=json.load(sys.stdin);target=Path(p['remote_package']);root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')
assert target.parent==root and target.name=='probability-probe-49a6b35-29822f9'
archive=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/transfers')/(target.name+'.tar.gz')
b=base64.b64decode(p['archive_base64']);assert hashlib.sha256(b).hexdigest()==p['archive_sha256']
assert not target.exists(),'Existing source package is never replaced'
archive.parent.mkdir(exist_ok=True,parents=True)
with archive.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
target.mkdir()
with tarfile.open(archive) as stream:
 for m in stream.getmembers():
  name=Path(m.name);assert not name.is_absolute() and '..' not in name.parts and (m.isdir() or m.isfile())
 stream.extractall(target)
manifest=target/'package-manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==p['manifest_sha256']
v=json.loads(manifest.read_bytes());actual={q.relative_to(target).as_posix() for q in target.rglob('*') if q.is_file() and q.name!='package-manifest.json'}
assert actual==set(v['files'])
for name,wanted in v['files'].items():
 q=target/name;assert not q.is_symlink() and hashlib.sha256(q.read_bytes()).hexdigest()==wanted,name
print(json.dumps(dict(status='SOURCE_AND_AUDIT_METADATA_ONLY_UPLOADED_NO_GPU_RELEASE_OR_SUBMISSION',remote_package=str(target),manifest_sha256=p['manifest_sha256'],archive_sha256=p['archive_sha256'],files=len(v['files']),gpu_submitted=False,tensor_assets_transferred=False),sort_keys=True))
'''
command='/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -c '+shlex.quote(remote)
result=subprocess.run(route(command),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr);sys.exit(result.returncode)

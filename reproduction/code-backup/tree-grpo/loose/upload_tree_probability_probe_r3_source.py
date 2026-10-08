"""Same-node source reuse and changed code/metadata upload only, no GPU action."""
import base64,gzip,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from remote import route
root=Path(__file__).resolve().parent
old=root/'probability-probe-49a6b35-29822f9-r2'
new=root/'probability-probe-49a6b35-29822f9-r3'
manifest=json.loads((new/'package-manifest.json').read_bytes())
prior=json.loads((old/'package-manifest.json').read_bytes())
copied=[];changed={}
for name,expected in dict(manifest['files'],**{'package-manifest.json':hashlib.sha256((new/'package-manifest.json').read_bytes()).hexdigest()}).items():
    if prior['files'].get(name)==expected:copied.append(name)
    else:changed[name]=base64.b64encode((new/name).read_bytes()).decode()
packet=dict(source=str(Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')/old.name),
    target=str(Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')/new.name),
    source_manifest_sha256=hashlib.sha256((old/'package-manifest.json').read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256((new/'package-manifest.json').read_bytes()).hexdigest(),copy_files=copied,changed=changed)
compressed=gzip.compress(json.dumps(packet,sort_keys=True).encode())
record=dict(status='SOURCE_DELTA_PREPARED_NO_GPU_ACTION',copied_same_node_source_files=len(copied),
    changed_code_or_metadata_files=len(changed),compressed_bytes=len(compressed),packet_sha256=hashlib.sha256(compressed).hexdigest(),
    source_manifest_sha256=packet['source_manifest_sha256'],manifest_sha256=packet['manifest_sha256'])
(root/'tree-probability-probe-owned/R3_CODE_UPLOAD_IDENTITY.json').write_text(json.dumps(record,indent=2)+'\n')
remote=r'''
import base64,gzip,hashlib,json,os,shutil,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
body=base64.b64decode(sys.stdin.read());p=json.loads(gzip.decompress(body));base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')
source=Path(p['source']);target=Path(p['target'])
assert source==base/'probability-probe-49a6b35-29822f9-r2' and target==base/'probability-probe-49a6b35-29822f9-r3'
assert hashlib.sha256((source/'package-manifest.json').read_bytes()).hexdigest()==p['source_manifest_sha256']
prior=json.loads((source/'package-manifest.json').read_bytes())
assert not target.exists(),'Existing immutable source revision is never replaced'
target.mkdir()
for name in p['copy_files']:
 n=Path(name);assert not n.is_absolute() and '..' not in n.parts
 s=source/n;d=target/n;assert s.resolve()==s and s.is_file() and hashlib.sha256(s.read_bytes()).hexdigest()==prior['files'][name]
 d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d)
for name,value in p['changed'].items():
 n=Path(name);assert not n.is_absolute() and '..' not in n.parts
 d=target/n;d.parent.mkdir(parents=True,exist_ok=True)
 with d.open('xb') as f:f.write(base64.b64decode(value));f.flush();os.fsync(f.fileno())
b=(target/'package-manifest.json').read_bytes();assert hashlib.sha256(b).hexdigest()==p['manifest_sha256'];m=json.loads(b)
actual={f.relative_to(target).as_posix() for f in target.rglob('*') if f.is_file() and f.name!='package-manifest.json'}
assert actual==set(m['files'])
for name,wanted in m['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==wanted
print(json.dumps(dict(status='SAME_NODE_SOURCE_REUSE_AND_CHANGED_CODE_METADATA_VERIFIED_NO_GPU_RELEASE',
 remote_package=str(target),manifest_sha256=p['manifest_sha256'],copied_same_node_source_files=len(p['copy_files']),
 changed_code_metadata_files=len(p['changed']),tensor_assets_transferred=False,gpu_submitted=False),sort_keys=True))
'''
command='/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -c '+shlex.quote(remote)
answer=subprocess.run(route(command),input=base64.b64encode(compressed),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
sys.stdout.buffer.write(answer.stdout);sys.stderr.buffer.write(answer.stderr);sys.exit(answer.returncode)

import hashlib,json,socket,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300')
archive=root/'transfers/recovery-502e9fa-f91c4f17.tar.gz'
target=root/'candidates/recovery-502e9fa-f91c4f17'
assert socket.gethostname().split('.')[0]=='h100-9'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='26c8b68f6f3e297ff1314c41d61ec2220e7fb9760a3e7fbdc06dc8377a1b2a4c'
target.mkdir(exist_ok=False)
with tarfile.open(archive) as t:
    for member in t.getmembers():
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    t.extractall(target)
raw=(target/'manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='f91c4f17a3a3454bb16a81e9ca400f4db07a56a0a27cf6cfac2c1814459c3087'
manifest=json.loads(raw)
for name,h in manifest['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==h,name
assert hashlib.sha256(Path(manifest['parent_manifest']['path']).read_bytes()).hexdigest()==manifest['parent_manifest']['sha256']
print(json.dumps(dict(status='FROZEN_RECOVERY_PACKAGE_PRESENT_NOT_SUBMITTED',path=str(target),files=len(manifest['files']),code=manifest['code_commit'])))

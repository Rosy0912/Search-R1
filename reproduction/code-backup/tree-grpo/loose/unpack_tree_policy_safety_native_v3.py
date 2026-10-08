from pathlib import Path
import tarfile,hashlib,json
base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates');archive=base/'tree-policy-safety-native-0980ec3-v3.tar.gz';target=base/'tree-policy-safety-native-0980ec3-v3'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4d776dc5ac0a62d171f1fa319dd79c7998886670c25530c16653d0c0bbd09a8d'
target.mkdir(exist_ok=False)
with tarfile.open(archive) as stream:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in stream.getmembers())
 stream.extractall(target)
manifest=target/'manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='1f2c5968db27770579fc550b19f7e7cea6ff0af66bfe9fa70dc05cfdcbd58724'
m=json.loads(manifest.read_text())
for name,sha in m['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==sha,name
print(json.dumps(dict(status='CODE_ONLY_VERIFIED_NOT_EXECUTED',source=m['code_commit'],files=len(m['files']))))

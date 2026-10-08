from pathlib import Path
import tarfile,hashlib,json
base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates');archive=base/'tree-policy-safety-native-0980ec3-v4.tar.gz';target=base/'tree-policy-safety-native-0980ec3-v4'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='64a64a5ed2a4dc7d753b3067ebc60696c39b540c894274fa54df371a7217b0d0'
target.mkdir(exist_ok=False)
with tarfile.open(archive) as stream:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in stream.getmembers())
 stream.extractall(target)
manifest=target/'manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='0935f213c61b70ea06faa2737ce31211693a4f9e848429a3b1c27ae7d76f4b1e'
m=json.loads(manifest.read_text())
for name,sha in m['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==sha,name
print(json.dumps(dict(status='CODE_ONLY_VERIFIED_NOT_EXECUTED',source=m['code_commit'],files=len(m['files']))))

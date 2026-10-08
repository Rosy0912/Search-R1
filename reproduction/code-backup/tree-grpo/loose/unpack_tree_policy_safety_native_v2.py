from pathlib import Path
import tarfile,hashlib,json
base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates');archive=base/'tree-policy-safety-native-0980ec3-v2.tar.gz';target=base/'tree-policy-safety-native-0980ec3-v2'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='a2a87e030b153662b853eacedcf346cc8ec0470baf8a4ec65a40100a8ddf440b'
target.mkdir(exist_ok=False)
with tarfile.open(archive) as stream:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in stream.getmembers())
 stream.extractall(target)
manifest=target/'manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='b9c2e07714a745e462d3678328ad4d25a3a83958c1683787cf63e1562768137c'
m=json.loads(manifest.read_text())
for name,sha in m['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==sha,name
print(json.dumps(dict(status='CODE_ONLY_VERIFIED_NOT_EXECUTED',source=m['code_commit'],files=len(m['files']))))

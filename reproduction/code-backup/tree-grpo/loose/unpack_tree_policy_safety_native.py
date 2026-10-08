from pathlib import Path
import tarfile,hashlib,json
base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates');archive=base/'tree-policy-safety-native-0980ec3.tar.gz';target=base/'tree-policy-safety-native-0980ec3'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='f51248e6feb948d713cde3b558684bd955724b8a4dffec685f7699f9ea21b6e3'
target.mkdir(exist_ok=False)
with tarfile.open(archive) as stream:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in stream.getmembers())
 stream.extractall(target)
manifest=target/'manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='ba9bba245a54e36c8026b21960eb4d644eefb97f930a1c2c2867269b8cbb1767'
m=json.loads(manifest.read_text())
for name,sha in m['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==sha,name
print(json.dumps(dict(status='CODE_ONLY_VERIFIED_NOT_EXECUTED',source=m['code_commit'],files=len(m['files']))))

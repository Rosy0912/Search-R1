import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'candidates';archive=base/'tree-fixed-two-296217a-v3.tar.gz';p=base/'tree-fixed-two-296217a-v3'
assert not p.exists(),'Do not replace candidate tools'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='0e0f086a0b6a958bba123933e5c7cee5fc6aa5f08c6fd99ff587a8ba5cda4dab'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]==p.name
 tar.extractall(base)
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='a8be5d5a0b8d86900f936ac112f2fb4bf5bbae0b16170801e542980d946fc686'
manifest=json.loads((p/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected
for name,expected in manifest['external_helpers'].items():assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'tools/start_tree_fixed_two_296217a_v3.sh').read_bytes()).hexdigest()=='682be03c9ea508e829337f31281dac60bc3aeef6fb77f801ff4ba64257a7995f'
for f in p.rglob('*'):
 if f.is_file():f.chmod(0o444)
print(json.dumps(dict(status='FIXED_TWO_TOOLS_ISOLATED_AND_VERIFIED',source=manifest['code_commit'],manifest='a8be5d5a0b8d86900f936ac112f2fb4bf5bbae0b16170801e542980d946fc686')))

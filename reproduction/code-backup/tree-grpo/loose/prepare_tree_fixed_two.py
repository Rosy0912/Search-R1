import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'candidates';archive=base/'tree-fixed-two-296217a.tar.gz';p=base/'tree-fixed-two-296217a'
assert not p.exists(),'Do not replace candidate tools'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='94f615c1845dcd6b01e31bdd5a0b0981b0964cc8823ee49ddb1c1315e5792d39'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]==p.name
 tar.extractall(base)
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='a7adf0e353b4ddc1d281d0b5ebf55b85630b243560f0dd3b16f6bd3394f683d1'
manifest=json.loads((p/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected
for name,expected in manifest['external_helpers'].items():assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'tools/start_tree_fixed_two_296217a.sh').read_bytes()).hexdigest()=='9a8804de6153d60b4e0192cdc085d42368684ce2b48a3793d84c02cc31dd8a43'
for f in p.rglob('*'):
 if f.is_file():f.chmod(0o444)
print(json.dumps(dict(status='FIXED_TWO_TOOLS_ISOLATED_AND_VERIFIED',source=manifest['code_commit'],manifest='a7adf0e353b4ddc1d281d0b5ebf55b85630b243560f0dd3b16f6bd3394f683d1')))

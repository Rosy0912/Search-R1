import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'candidates';archive=base/'tree-fixed-two-296217a-v2.tar.gz';p=base/'tree-fixed-two-296217a-v2'
assert not p.exists(),'Do not replace candidate tools'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4da33ec0f2ad1c8e665c600f0f9002f2e2fc12a3e9c2009f373d29264cd84e1f'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]==p.name
 tar.extractall(base)
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='2397533d886ec8f0ce3cd1786fbc32c527b37b26eadc3e1b42814ba66f726a9d'
manifest=json.loads((p/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected
for name,expected in manifest['external_helpers'].items():assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'tools/start_tree_fixed_two_296217a_v2.sh').read_bytes()).hexdigest()=='7b330925800e62a52f6919df858daebbd4dfdd63c1af0c76bfd0de21bc479183'
for f in p.rglob('*'):
 if f.is_file():f.chmod(0o444)
print(json.dumps(dict(status='FIXED_TWO_TOOLS_ISOLATED_AND_VERIFIED',source=manifest['code_commit'],manifest='2397533d886ec8f0ce3cd1786fbc32c527b37b26eadc3e1b42814ba66f726a9d')))

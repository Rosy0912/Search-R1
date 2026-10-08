import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'candidates';archive=base/'tree-fixed-five-296217a.tar.gz';p=base/'tree-fixed-five-296217a'
assert not p.exists(),'Do not replace candidate tools'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='b2e6f5fff3499df09313a2f021235a90433b13c27be1b129d9e488151071654f'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]==p.name
 tar.extractall(base)
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='8f3fb72fc21037d4d629026be9db0cac00c24b00df21a45982d4a63a0fd6710b'
manifest=json.loads((p/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected
for name,expected in manifest['external_helpers'].items():assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'tools/start_tree_fixed_five_296217a.sh').read_bytes()).hexdigest()=='b5f2932c507d75defd83b9b84e203e3634f0f0b1d94daf120377e9c59b0e624a'
for f in p.rglob('*'):
 if f.is_file():f.chmod(0o444)
print(json.dumps(dict(status='FIXED_TWO_TOOLS_ISOLATED_AND_VERIFIED',source=manifest['code_commit'],manifest='8f3fb72fc21037d4d629026be9db0cac00c24b00df21a45982d4a63a0fd6710b')))

import shutil
assert hashlib.sha256((root/'tools/check_fixed_five_afterany_296217a.py').read_bytes()).hexdigest()=='a6a73cf43a659d55402d56005d975749a5766536ceec54bffa8e0f364f3b96d3'
review=json.loads((p/'ACTUAL3095_RECEIPT.json').read_text())
for item in review['small_review_bundle_mapping']:
 target=Path(item['path']);assert str(target).startswith(str(root/'logs/tree-fixed-two-3095-review')+'/')
 source=p/Path(item['local_path']).name;assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
 target.parent.mkdir(exist_ok=True)
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
 else:shutil.copyfile(source,target)
 target.chmod(0o444)
(root/'tools/check_fixed_five_afterany_296217a.py').chmod(0o444)
(root/'tools/start_tree_fixed_five_296217a.sh').chmod(0o444)
print('FIVE_AND_SMALL_PRIOR_REVIEW_READY_NO_GPU_DISPATCH')

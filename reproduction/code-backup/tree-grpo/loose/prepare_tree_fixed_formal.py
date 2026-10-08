import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');base=root/'candidates';archive=base/'tree-fixed-formal-296217a.tar.gz';p=base/'tree-fixed-formal-296217a'
assert not p.exists(),'Do not replace candidate tools'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='01bdbd65b4c59079c58764f61a10a0fd03cce27d305540b1aadc8db59dd202d0'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]==p.name
 tar.extractall(base)
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='f66afebc025db70c2cd9391442fae6834722de95a6dcafc030416d60dee4e911'
manifest=json.loads((p/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected
for name,expected in manifest['external_helpers'].items():assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'tools/start_tree_fixed_formal_296217a.sh').read_bytes()).hexdigest()=='1e34668c228c48642a8610a20a4ec98f875bd826e41f05e1451acb7b33843b06'
for f in p.rglob('*'):
 if f.is_file():f.chmod(0o444)
print(json.dumps(dict(status='FIXED_TWO_TOOLS_ISOLATED_AND_VERIFIED',source=manifest['code_commit'],manifest='f66afebc025db70c2cd9391442fae6834722de95a6dcafc030416d60dee4e911')))

import shutil
assert hashlib.sha256((root/'tools/check_fixed_formal_afterany_296217a.py').read_bytes()).hexdigest()=='19aac6802322dbaf5af370f13b890813456cbd940d1ab14dd5f1c0972acd6091'
review=json.loads((p/'ACTUAL3103_RECEIPT.json').read_text())
for item in review['small_review_bundle_mapping']:
 target=Path(item['path']);assert str(target).startswith(str(root/'logs/tree-fixed-five-3103-review')+'/')
 source=p/Path(item['local_path']).name;assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
 target.parent.mkdir(exist_ok=True)
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
 else:shutil.copyfile(source,target)
 target.chmod(0o444)
(root/'tools/check_fixed_formal_afterany_296217a.py').chmod(0o444)
(root/'tools/start_tree_fixed_formal_296217a.sh').chmod(0o444)
print('FIVE_AND_SMALL_PRIOR_REVIEW_READY_NO_GPU_DISPATCH')

(root/'tools/summarize_tree_fixed_formal_296217a.py').chmod(0o444)

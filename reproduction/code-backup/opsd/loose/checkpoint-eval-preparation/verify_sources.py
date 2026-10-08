import hashlib,json,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
manifest=json.loads((root/'tools/checkpoint-eval/source-manifest.json').read_text())
assert manifest['source_commit']=='e43fbfa73088ed70ed3efef79ba561507c0afc33'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
for directory,key in (('runtime','runtime_sources'),('checkpoint-eval','eval_sources')):
 for name,sha in manifest[key].items():
  assert Path(name).name==name
  p=root/'tools'/directory/name
  assert p.is_file() and p.resolve().is_relative_to(root.resolve())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,(directory,name)
print('CHECKPOINT_EVAL_PINNED_SOURCES_VERIFIED',flush=True)

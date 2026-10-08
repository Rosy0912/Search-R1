import hashlib,json
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-native-numeric-3063';files={}
for p in sorted(run.rglob('*')):
 if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.log'):
  b=p.read_bytes();files[str(p.relative_to(run))]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode())
log=root/'logs/tree-native-numeric-3063.out';b=log.read_bytes();files['slurm-output.txt']=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode())
p=root/'candidates/tree-native-numeric-296217a/manifest.json';manifest=p.read_bytes()
print(json.dumps(dict(run=str(run),manifest_sha256=hashlib.sha256(manifest).hexdigest(),files=files)))

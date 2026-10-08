from pathlib import Path
import sys,json,hashlib,base64
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');p=root/'scheduler-head'
assert not root.exists() and not root.is_symlink()
p.mkdir(parents=True);assert p.resolve()==p and root.resolve()==root
items=json.load(sys.stdin)['files'];assert set(items)=={'submit.sh','hardware.sbatch','formal300.sbatch','approved-launch-hashes.json','watch-terminal.py'}
for n,v in items.items():
 raw=base64.b64decode(v['base64'],validate=True);assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
 with(p/n).open('xb') as f:f.write(raw)
 (p/n).chmod(0o755)
m=json.loads((p/'approved-launch-hashes.json').read_text())
for n in ['submit.sh','hardware.sbatch','formal300.sbatch']:assert hashlib.sha256((p/n).read_bytes()).hexdigest()==m[n]
print('R3_HEAD_UNIQUE_LEDGER_PREPARED_NO_JOB')

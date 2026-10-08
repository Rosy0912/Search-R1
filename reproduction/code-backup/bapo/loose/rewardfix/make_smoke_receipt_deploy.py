from pathlib import Path
import base64
mapping={n:n for n in []}
paths=['outputs/bapo-rewardfix-smoke3069-independent-review.md','outputs/bapo-rewardfix-smoke3069-independent-cost.json','outputs/bapo-rewardfix-smoke3069-independent-recomputed.json','work/evidence/rewardfix-smoke3069-independent-current-hashes.json','outputs/bapo-rewardfix-smoke3069-acceptance.json']
records={Path(p).name:base64.b64encode(Path(p).read_bytes()).decode() for p in paths};records['smoke2-3069-acceptance.json']=records['bapo-rewardfix-smoke3069-acceptance.json']
s='''import pathlib,base64,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')
for name,data in RECORDS.items():
 p=root/'logs'/name;b=base64.b64decode(data)
 assert not p.is_symlink()
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
subprocess.run([str(root/'env/bin/python'),str(root/'tools/verify_smoke_receipt-gpu64.py'),'3069'],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
'''.replace('RECORDS',repr(records))
Path('work/rewardfix/deploy-smoke3069-receipt.py').write_text(s)

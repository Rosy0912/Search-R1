import json
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009')
base=Path('/dataset/shaoshitong/search-r1-rl')
out=[]
for r in base.glob('bapo-aligned-cohost-node11-v*-20261009'):
 for p in r.glob('runs/*3356/**/native-initial-rank_0.json'):
  v=json.loads(p.read_text());out.append({'path':str(p),'process':v.get('native_process')})
print(json.dumps(out))

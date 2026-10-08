python3 - <<'CHECK'
import json
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-n')
for n in ['training.log','worker-training.log']:
 rows=[]
 for l in (r/n).read_text(errors='replace').splitlines():
  if 'COSEARCH_ACTOR_INITIALIZATION_JSON=' in l:rows.append(json.JSONDecoder().raw_decode(l.split('COSEARCH_ACTOR_INITIALIZATION_JSON=',1)[1])[0])
 print(json.dumps(dict(file=n,rows=rows)))
print((r/'actor-gpu-ownership-first.json').read_text())
CHECK

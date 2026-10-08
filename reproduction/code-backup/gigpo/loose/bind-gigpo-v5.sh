set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
export AGENT_ROOT="$ROOT"
python3 - <<'PY'
from pathlib import Path
import json,hashlib,os
r=Path(os.environ['AGENT_ROOT'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((r/'logs/deployment-binding-27c989b-v4.json').read_text())
for x in b['evidence']:
 p=Path(x['path'])
 if p==r/'tools/owned_runtime.py':x['sha256']=sha(p);assert x['sha256']=='429d5b6849c8c3deec9102abf03c41848c4dfca119d0eff2a006f28670fcf9d3'
 else:assert sha(p)==x['sha256'],p
review=r/'logs/own-cleanup-grace-independent-review-v7.json';rev=json.loads(review.read_text());assert rev['status']=='REVIEWED'
log=r/'logs/cleanup-cpu-2963.log';assert '"status": "PASS"' in log.read_text()
for p in [review,log,r/'logs/owned_runtime-2949-original.py',r/'tools/cleanup-real-cpu.py']:
 b['evidence'].append(dict(path=str(p),sha256=sha(p)))
b['scope']='Reviewed same-node GPU deployment, originalPIDfd guard and bounded60sec own cleanup grace; newactual5+64validation+HF still required; previous2949FAILED preserved'
p=r/'logs/deployment-binding-27c989b-v5.json'
with p.open('x') as f:json.dump(b,f,indent=2)
print('BINDING_SHA='+sha(p))
PY

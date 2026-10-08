set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/grpo-control
export AGENT_ROOT="$ROOT"
python3 - <<'PY'
from pathlib import Path
import os,json,hashlib
r=Path(os.environ['AGENT_ROOT'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=json.loads((r/'logs/deployment-binding-907a736-v1.json').read_text())
for x in b['evidence']:
 p=Path(x['path'])
 if p==r/'tools/grpo-short-run.py':x['sha256']=sha(p);assert x['sha256']=='478bba35c53304c8e9f386e45bf009cc54efc352b0e98e3a841fbaddb398f568'
 else:assert sha(p)==x['sha256'],p
review=r/'logs/grpo-checkpoint-gate-independent-review-v8.json';assert sha(review)=='03b5d1ae4bcc703f28b216a059c2433f363990734be3d1deab23784578ea494e';assert json.loads(review.read_text())['status']=='REVIEWED'
for p in [review,r/'logs/grpo-short-run-2973-original.py']:
 b['evidence'].append(dict(path=str(p),sha256=sha(p)))
b['scope']='GRPO deploymentv6+cleanupv7+GRPO no-controller-state checkpoint gatev8; original2973FAILED retained. Require newactual2/fullHF/cleanexit/indepPASS.'
p=r/'logs/deployment-binding-907a736-v2.json'
with p.open('x') as f:json.dump(b,f,indent=2)
print('GRPO_V2_SHA='+sha(p))
PY

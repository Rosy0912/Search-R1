set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/grpo-control
export AGENT_ROOT="$ROOT"
cd "$ROOT/repo"
test "$(git rev-parse HEAD)" = 907a73661041980fa89c6a06fae3449aa77609ed
test -z "$(git status --porcelain)"
python3 - <<'PY'
from pathlib import Path
import json,hashlib,os
r=Path(os.environ['AGENT_ROOT']);g=r.parent/'gigpo'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=r/'logs/grpo-deployment-ready-independent-review-v6.json';assert sha(review)=='e9180a140c848060f8857b6a9d14db758b723fcdb9c9a6d0a8ffa77548abf5f0'
v=json.loads(review.read_text());assert v['status']=='REVIEWED' and v['deployment_ready'];items=[]
for x in v['evidence']:
 p=Path(x['path']);actual=sha(p)
 if p==r/'tools/owned_runtime.py':assert actual=='1b55e95f30020ac3384bc86fe62b21c5877569fc7db61e90464b7aee5daba92f'
 else:assert actual==x['sha256'],p
 items.append(dict(path=str(p),sha256=actual))
v7=r/'logs/own-cleanup-grace-independent-review-v7.json';assert sha(v7)=='df68d487c9710a9ef06bd9b967d1d0e9c2b2059ce1d3745be49ade28e7f1a9a1'
log=g/'logs/cleanup-cpu-2963.log';assert '"status": "PASS"' in log.read_text()
for p in [review,v7,log,g/'logs/hardware-2903.log',g/'logs/hardware-2903.exit',g/'runs/hardware-2903/resource-samples.jsonl',g/'runs/hardware-2903/actor-entropy-compile.json',g/'logs/gpu-guard-identity-independent-review-v5.json']:
 items.append(dict(path=str(p),sha256=sha(p)))
record=dict(status='REVIEWED',code_commit=v['code_commit'],public_commit=v['public_commit'],evidence=items,native_cpu_pass=172,reviewer='review_common',scope='GRPO deployment v6 and owncleanup v7; sharedhardware2903 capability only, no GRPO actualGPU acceptance. NewrealHTTP/2updates/HF gates required.',hardware_reuse_scope=v['hardware_reuse_scope'],superseded_owned_runtime_sha256='a3a03013a22f800a49fab67ec48af9f302aa9933ee8831760fd6dab43f2bc5d6')
p=r/'logs/deployment-binding-907a736-v1.json'
with p.open('x') as f:json.dump(record,f,indent=2)
print('GRPO_BINDING_SHA='+sha(p))
PY

from pathlib import Path
import json,hashlib
C=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/tree-f994af3-aa9e3161');m=json.loads((C/'manifest.json').read_bytes());out=[]
for name in ['source/repo/search_r1/baselines/objectives.py','source/repo/verl/trainer/ppo/core_algos.py']:
 b=(C/name).read_bytes();h=hashlib.sha256(b).hexdigest();assert h==m['files'][name];out.append(dict(path=str(C/name),sha256=h))
print(json.dumps(dict(status='FROZEN_OBJECTIVE_SOURCE_SHA_PASS',proof=out)))

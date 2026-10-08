set -euo pipefail
CUDA_VISIBLE_DEVICES='' /dataset/shaoshitong/search-r1-rl/cosearch/author-native/env/bin/python - <<'PY'
import sys,json,hashlib,collections
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');r=root/'runs/author-native-continue-300-r';sys.path[:0]=[str(root/'repo'),str(root/'repo/verl')]
import torch
result=[]
for p in sorted((r/'evidence/raw-rollouts').glob('*ranker-filter*.pt')):
 if int(p.stem.rsplit('-',1)[-1])<2880149136671638:continue
 x=torch.load(p,map_location='cpu',weights_only=False);d=dict(file=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 if p.name.startswith('pre-ranker-filter'):
  a=x['args'];d.update(main_rows=a[1] if len(a)>1 else None,nested_groups=len(a[0]),raw_ranker_outputs=sum(len(g) for g in a[0]),group_sizes=dict(collections.Counter(len(g) for g in a[0])))
 else:d.update(retained_outputs=len(x[0]),metrics=x[1])
 result.append(d)
assert len(result)==32 and not torch.cuda.is_initialized()
out=dict(run=r.name,step=11,files=result,CUDA_initialized=False,RL_updates=0)
p=r/'evidence/step11-raw-diagnosis.json';p.write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps(out,default=str),flush=True)
PY

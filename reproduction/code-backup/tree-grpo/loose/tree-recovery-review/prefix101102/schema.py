import json,gzip
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output')
for p in [next((root/'trajectories/outer-000101').glob('*.gz')),next((root/'tree-audit').glob('step_101_rollout_*.json'))]:
 if p.suffix=='.gz':
  with gzip.open(p,'rt') as f:x=json.loads(next(f))
  print('metadata',x['metadata']);print('keys',x.keys())
 else:
  x=json.loads(p.read_bytes());print('leaffirst',x.get('leaves',[None])[0]);print('groupfirst',str(x.get('question_groups'))[:2000])

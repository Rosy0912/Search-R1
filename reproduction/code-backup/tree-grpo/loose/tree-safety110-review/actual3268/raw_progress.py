from pathlib import Path
import json,time,hashlib
R=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268/output/sampling-consistency');rows=[]
for p in sorted(R.glob('rank-00-call-*.json')):
 b=p.read_bytes();j=json.loads(b);rows.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'call':j['generation_call'],'rows':j['local_rows'],'saved':j['saved_rows'],'decision':j['decision'],'distribution':j['distribution']})
print(json.dumps({'epoch':time.time(),'rank0_only_prefix':rows,'complete8rank_acceptance':False}))

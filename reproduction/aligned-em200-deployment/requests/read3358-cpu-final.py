import json,datetime
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009/evidence/first5-cpu3358-20261009-1806/cpu-evidence.json')
if p.exists():print(p.read_text())
else:print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pending':True,'path':str(p)}))

import json,datetime
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/evidence/first5-cpu3359-20261009-1835/cpu-evidence.json')
if p.exists():print(p.read_text())
else:print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pending':True,'path':str(p)}))

from pathlib import Path
import json
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356')
for p in sorted(r.rglob('native-initial*.json')):
 d=json.loads(p.read_text());print(json.dumps({'path':str(p),'data':d}))

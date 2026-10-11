from pathlib import Path
import json
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');run=Path(json.loads((r/'scheduler-head/eval150-submit-receipt.json').read_text())['run']);p=run/'resource-watch.jsonl';d=json.loads(p.read_text().splitlines()[-1]);print(json.dumps(d))

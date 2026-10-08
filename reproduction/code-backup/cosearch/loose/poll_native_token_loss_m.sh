set -euo pipefail
python3 - <<'PY'
import json,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');r=root/'runs/author-native-token-loss-probe-m'
x=dict(run=r.name,checked_at=time.time(),queue=subprocess.check_output(['squeue','-h','--name=cosearch-native-hf-probe','-o','%i %T %N'],text=True),files={})
for p in [root/'logs/native-token-loss-m.log',r/'result.json',r/'driver-exit.json']:
 if p.exists():x['files'][p.name]=p.read_text()[-14000:] if p.suffix=='.log' else json.loads(p.read_text())
print(json.dumps(x))
PY

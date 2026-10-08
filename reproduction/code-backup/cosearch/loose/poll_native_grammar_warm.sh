set -euo pipefail
python3 - <<'POLL'
import json,subprocess,time,hashlib
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');run=root/'runs/author-native-grammar-warm-start-v1'
v=dict(run=run.name,checked_at=time.time(),queue=subprocess.check_output(['squeue','-h','--name=cosearch-native-warm','-o','%i %T %N'],text=True),files={})
for p in [root/'logs/native-grammar-warm-v1.log',root/'logs/native-warm-start-failure.json',run/'driver-exit.json',run/'leases.json',run/'warm-start/manifest.json',run/'warm-start/native-supervised-parity.json']:
 if p.exists():
  b=p.read_bytes();v['files'][str(p.relative_to(root))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode(errors='replace')[-12000:] if p.suffix=='.log' else b.decode(errors='replace'))
print(json.dumps(v))
POLL

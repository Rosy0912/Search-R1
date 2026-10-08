set -euo pipefail
python3 - <<'POLL'
import json,hashlib,subprocess,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-raw-five-e')
v=dict(run=r.name,job=3050,checked_at=time.time(),queue=subprocess.check_output(['squeue','-h','--name=cosearch-author-five','-o','%i %T %N'],text=True),files={})
for name in ['driver-exit.json','startup-failure.json','launch.json','training.log','retriever.log','ray-head.log']:
 p=r/name
 if p.exists():
  b=p.read_bytes();v['files'][name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode(errors='replace')[-16000:] if name.endswith('.log') else b.decode())
v['workers']={p.name:p.read_text() for p in sorted((r/'evidence/workers').glob('*.jsonl'))}
v['controllers']={p.name:p.read_text()[-20000:] for p in sorted((r/'evidence').glob('controller-*.jsonl'))}
print(json.dumps(v))
POLL

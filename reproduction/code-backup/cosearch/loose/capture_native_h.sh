set -euo pipefail
python3 - <<'CAPTURE'
import json,hashlib,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-gpu-routing-probe-h')
files=['driver-exit.json','launch.json','evidence/routing-old_early.json','evidence/routing-deferred.json','evidence/routing-cpu-reuse.json']
files += [str(p.relative_to(r)) for p in sorted((r/'evidence/workers').glob('*.jsonl'))]
v={'run':r.name,'job':3048,'queue':subprocess.check_output(['squeue','-h','--name=cosearch-author-five','-o','%i %T %N'],text=True),'files':{}}
for n in files:
 p=r/n
 if p.exists():v['files'][n]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(v))
CAPTURE

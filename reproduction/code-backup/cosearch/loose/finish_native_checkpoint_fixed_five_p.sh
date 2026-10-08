set -euo pipefail
python3 - <<'PY'
import json,hashlib,time,shutil
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');t=r/'tools';stage=r/'checkpoint-fixed-tools-p';old=r/'runs/author-native-phase-aware-warm-five-m/tools'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ready=json.loads((old/'preparation-ready.json').read_text());gate=json.loads((t/'cpu-launch-gate.json').read_text())
assert gate['passed'] and gate['argv_count']==102 and gate['data']['train']['rows']==166222 and gate['data']['validation']['rows']==64
changed={'checkpoint_observer.py','warm-initialization.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert sha(t/n)==h,n
for n in changed|{'checkpoint-digest-correction.json'}:ready['tools_sha256'][n]=sha(t/n)
ready.update(created_at=time.time(),run='author-native-checkpoint-fixed-five-p',initialization='M same actorv4/rankerv2/112SFT/freshRLAdam; reviewed checkpoint digest OmegaConf compatibility only')
assert ready['cpu_preparation_passed'] and not ready['actual_gpu_resources_tested']
(t/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
for n in ['preparation-ready.json','cpu-launch-gate.json','resolved-author.yaml']:shutil.copy2(t/n,stage/n)
print(json.dumps(dict(run=ready['run'],ready_sha256=sha(t/'preparation-ready.json'),tools_sha256=ready['tools_sha256'],cpu_launch_gate=gate,accepted=False)),flush=True)
PY

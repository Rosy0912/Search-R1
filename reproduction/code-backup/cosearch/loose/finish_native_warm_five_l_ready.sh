set -euo pipefail
python3 - <<'VERIFY'
import json,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
for name in ['cosearch-author-five','cosearch-native-warm','cosearch-native-hf-probe']:assert not subprocess.check_output(['squeue','-h','--name='+name,'-o','%i'],text=True).strip()
assert json.loads((r/'tools/warm-initialization.json').read_text())['run']=='author-native-warm-five-l'
c=json.loads((r/'tools/cpu-launch-gate.json').read_text());assert c['passed'] and c['configuration_composed'] and c['argv_count']==102 and c['model_path']==str(r/'runs/author-native-grammar-actor-continue-v4/warm-start/actor')
assert c['data']['train']['rows']==166222 and c['data']['validation']['rows']==64
assert not (r/'author-native-warm-five-l-launch-claim.json').exists()
VERIFY
python3 - <<'READY'
import json,hashlib,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
p=root/'tools/preparation-ready.json';ready=json.loads(p.read_text())
assert json.loads((root/'tools/cpu-launch-gate.json').read_text())['passed']
for n,expected in ready['tools_sha256'].items():
 if n not in ['author-cli.json','warm-initialization.json']:assert hashlib.sha256((root/'tools'/n).read_bytes()).hexdigest()==expected,n
ready['tools_sha256']['author-cli.json']=hashlib.sha256((root/'tools/author-cli.json').read_bytes()).hexdigest()
ready['tools_sha256']['warm-initialization.json']=hashlib.sha256((root/'tools/warm-initialization.json').read_bytes()).hexdigest()
ready['created_at']=time.time();ready['initialization']='native actorv4/rankerv2;112SFT initialization; fresh RL Adam'
p.write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(ready))
READY

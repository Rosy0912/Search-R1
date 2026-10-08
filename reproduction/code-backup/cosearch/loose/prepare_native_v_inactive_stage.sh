set -euo pipefail
python3 - <<'REMOTE_U'
from pathlib import Path
import json,hashlib,subprocess,shutil
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');src=r/'runs/author-native-continue-300-u/tools';s=r/'continue-300-tools-v'
assert not s.exists() and not (r/'runs/author-native-continue-300-v').exists() and not (r/'author-native-continue-300-v-launch-claim.json').exists()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert hashlib.sha256((src/'preparation-ready.json').read_bytes()).hexdigest()=='9d387aa2d25c890332fa17f01d18ed49ee42c83925fc23d5db107a2205a8a95d'
ready=json.loads((src/'preparation-ready.json').read_text())
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h,n
shutil.copytree(src,s);(s/'preparation-ready.json').unlink();print('inactive U stage copied frozen T tools; source not modified')
REMOTE_U

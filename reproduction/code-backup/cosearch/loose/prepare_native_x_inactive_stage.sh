set -euo pipefail
python3 - <<'REMOTE_U'
from pathlib import Path
import json,hashlib,subprocess,shutil
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');src=r/'runs/author-native-continue-300-w/tools';s=r/'continue-300-tools-x'
assert not s.exists() and not (r/'runs/author-native-continue-300-x').exists() and not (r/'author-native-continue-300-x-launch-claim.json').exists()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert hashlib.sha256((src/'preparation-ready.json').read_bytes()).hexdigest()=='184a390530e4fc87100d48869688ed5b4aac99f803f5594721e756d915ef9106'
ready=json.loads((src/'preparation-ready.json').read_text())
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h,n
shutil.copytree(src,s);(s/'preparation-ready.json').unlink();print('inactive U stage copied frozen T tools; source not modified')
REMOTE_U

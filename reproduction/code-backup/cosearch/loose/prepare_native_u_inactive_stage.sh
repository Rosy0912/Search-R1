set -euo pipefail
python3 - <<'REMOTE_U'
from pathlib import Path
import json,hashlib,subprocess,shutil
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');src=r/'runs/author-native-continue-300-t/tools';s=r/'continue-300-tools-u'
assert not s.exists() and not (r/'runs/author-native-continue-300-u').exists() and not (r/'author-native-continue-300-u-launch-claim.json').exists()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert hashlib.sha256((src/'preparation-ready.json').read_bytes()).hexdigest()=='4fbf415779151ee020ab5275d156fdffa97c12e24fc776c0d0ab6d9341cbcbce'
ready=json.loads((src/'preparation-ready.json').read_text())
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h,n
shutil.copytree(src,s);(s/'preparation-ready.json').unlink();print('inactive U stage copied frozen T tools; source not modified')
REMOTE_U

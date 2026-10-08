set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil,subprocess,time
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-u';t=r/'tools'
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert hashlib.sha256((s/'preparation-ready.json').read_bytes()).hexdigest()=='9d387aa2d25c890332fa17f01d18ed49ee42c83925fc23d5db107a2205a8a95d'
ready=json.loads((s/'preparation-ready.json').read_text())
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
for p in s.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.sh','.yaml']:shutil.copy2(p,t/p.name)
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((t/n).read_bytes()).hexdigest()==h,n
assert (s/'preparation-ready.json').read_bytes()==(t/'preparation-ready.json').read_bytes()
print(json.dumps(dict(run=ready['run'],ready_sha256=hashlib.sha256((t/'preparation-ready.json').read_bytes()).hexdigest(),deployed_readback_equal=True,tools_sha256=ready['tools_sha256'],checked_at=time.time())),flush=True)
PY

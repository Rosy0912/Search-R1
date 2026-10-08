import subprocess
from pathlib import Path
command=r'''python3 - <<'INNER'
import hashlib,importlib.metadata,json,pathlib
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/env/lib/python3.10/site-packages')
files=list(pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/env/lib').glob('python*/site-packages/vllm/envs.py'));assert len(files)==1
p=files[0];text=p.read_text()
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cache_definitions=[line for line in text.splitlines() if any(key in line for key in ('VLLM_CACHE_ROOT','XDG_CACHE_HOME'))]),indent=2))
INNER'''
r=subprocess.run(['python3','work/remote_topd.py'],input=command,text=True,capture_output=True,timeout=90)
Path('outputs/topd-vllm063-cache-source-readonly.log').write_text(r.stdout+r.stderr)
print(r.stdout+r.stderr);raise SystemExit(r.returncode)

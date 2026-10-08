import sys,json,shlex,subprocess
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
code=r'''import inspect,json,hashlib
from pathlib import Path
from tensordict import TensorDict
from tensordict.base import TensorDictBase
out={}
for cls,n in ((TensorDictBase,'cuda'),(TensorDict,'to')):
 f=getattr(cls,n);p=Path(inspect.getsourcefile(f));out[n]=dict(signature=str(inspect.signature(f)),source=inspect.getsource(f),file_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(out))
'''
r=subprocess.run(control.route('h100-8','/dataset/shaoshitong/Search-G1-common8/env/bin/python -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
Path('/Users/shaoshitong/Documents/Search-G1/preparation/tensordict-device-source.json').write_text(r.stdout);p=json.loads(r.stdout);print(p['cuda']['signature']);print(p['cuda']['source']);print(p['to']['source'][:1600])

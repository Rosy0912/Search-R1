import sys,subprocess,shlex,json,base64,uuid
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
repo=Path('/Users/shaoshitong/Documents/Search-G1/repo')
files=['verl/utils/rope_precision.py','verl/utils/fsdp_utils.py','verl/workers/fsdp_workers.py','reproduction/common8/test_rope_precision.py']
encoded={f:base64.b64encode((repo/f).read_bytes()).decode() for f in files}
base='/dataset/shaoshitong/g1-common8-rope-fix-cpu-'+uuid.uuid4().hex[:12]
code='import shutil,base64,subprocess,json,hashlib;from pathlib import Path\nb=Path('+repr(base)+');shutil.copytree("/dataset/shaoshitong/g1-common8-device-fix-cpu-d5a7a2d82f65",b)\nfiles='+repr(encoded)+'\nfor f,v in files.items():\n p=b/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-m","unittest","reproduction.common8.test_rope_precision"],cwd=b,capture_output=True,text=True,timeout=120)\nprint(json.dumps(dict(base=str(b),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,sha256={f:hashlib.sha256((b/f).read_bytes()).hexdigest() for f in files})))'
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=160);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);Path('/Users/shaoshitong/Documents/Search-G1/preparation/rope-fix-target-cpu.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))

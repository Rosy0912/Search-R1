"""Hash preserved retriever environment/source; read-only, no GPU operations."""
import importlib.util,subprocess,shlex
from pathlib import Path
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
code=r"""from pathlib import Path
import os,json,hashlib,time
b=Path('/dataset/from_h100-16/shaoshitong/search-r1-rl/cosearch');n=0
for root in ['env','repo']:
 for d,ds,fs in os.walk(b/root,followlinks=False):
  ds[:]=sorted(x for x in ds if x not in ['.git','__pycache__'])
  for name in list(ds)+sorted(fs):
   p=Path(d)/name;r=str(p.relative_to(b))
   if p.is_symlink():print(json.dumps({'kind':'symlink','path':r,'target':os.readlink(p)}),flush=True)
   elif p.is_file():
    h=hashlib.sha256()
    with p.open('rb') as f:
     for q in iter(lambda:f.read(16777216),b''):h.update(q)
    print(json.dumps({'kind':'file','path':r,'bytes':p.stat().st_size,'sha256':h.hexdigest()}),flush=True);n+=1
print(json.dumps({'kind':'complete','files':n,'checked_epoch':time.time()}),flush=True)
"""
out=Path(__file__).resolve().parents[1]/'outputs/evidence/native-x-hpc-retriever-supplement-manifest.jsonl'
with out.open('w') as f, out.with_suffix('.stderr').open('w') as e:
 r=subprocess.run(c.route('h100-11','python3 -c '+shlex.quote(code)),stdout=f,stderr=e,timeout=1800)
if r.returncode:raise SystemExit(r.returncode)

import sys,json,subprocess,shlex,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base=json.loads((prep/'attempt2-preparation.json').read_text())['base']
code='import json,time;from pathlib import Path\nb=Path('+repr(base)+')\n'+r'''
p=b/'logs/model-initial.log';out=dict(checked_unix=time.time(),tail=p.read_text(errors='replace')[-16000:] if p.exists() else None)
f=b/'logs/model-smoke-initial/complete.json'
if f.exists():out['complete']=json.loads(f.read_text())
print(json.dumps(out))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'attempt2-model-observation.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))

import sys,subprocess,shlex,json,base64,uuid,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');repo=prep.parent/'repo'
files=['search_g1/policy_safety.py','search_g1/test_policy_safety.py','search_g1/gradient_clipping.py','verl/trainer/ppo/core_algos.py','verl/workers/actor/dp_actor.py','verl/trainer/config/ppo_trainer.yaml','reproduction/common8/recipe.json','reproduction/common8/worker_evidence.py']
claim=dict(base='/dataset/shaoshitong/g1-common8-safety-cpu-'+uuid.uuid4().hex[:12],source_sha256={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files})
(prep/'safety-target-cpu-claim.json').write_text(json.dumps(claim,indent=2))
encoded={f:base64.b64encode((repo/f).read_bytes()).decode() for f in files}
code='import shutil,base64,subprocess,json,hashlib;from pathlib import Path\nb=Path('+repr(claim['base'])+');shutil.copytree("/dataset/shaoshitong/g1-common8-rope-fix-cpu-2f7413107aa6",b)\nfiles='+repr(encoded)+'\nfor f,v in files.items():\n p=b/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-m","unittest","-v","search_g1.test_policy_safety"],cwd=b,capture_output=True,text=True,timeout=90)\nx=dict(base=str(b),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,sha256={f:hashlib.sha256((b/f).read_bytes()).hexdigest() for f in files});(b/"safety-cpu-result.json").write_text(json.dumps(x,indent=2));print(json.dumps(x))'
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=150);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(prep/'safety-target-cpu-result.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))

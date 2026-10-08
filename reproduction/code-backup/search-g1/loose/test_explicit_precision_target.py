import sys,json,subprocess,shlex,uuid,base64,gzip,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
root=Path('/Users/shaoshitong/Documents/Search-G1');repo=root/'repo';prep=root/'preparation'
base='/dataset/shaoshitong/g1-common8-explicit-precision-cpu-'+uuid.uuid4().hex[:12]
names=['search_g1/compute_precision.py','search_g1/test_compute_precision.py','verl/workers/actor/dp_actor.py','reproduction/common8/configuration.py','reproduction/common8/recipe.json','verl/trainer/config/ppo_trainer.yaml','reproduction/common8/model_smoke.py','reproduction/common8/test_model_smoke.py','reproduction/common8/test_configuration.py','reproduction/common8/precision_evidence.py','reproduction/common8/test_precision_evidence.py','reproduction/common8/safety_calibration_evidence.py','reproduction/common8/test_safety_calibration_evidence.py']
values={n:base64.b64encode((repo/n).read_bytes()).decode() for n in names};hashes={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}
(prep/'explicit-precision-cpu-launch.json').write_text(json.dumps(dict(base=base,source_sha256=hashes,scope='CPU only, isolated source; no native GPU or training writes'),indent=2))
code='import os,json,shutil,base64,hashlib,subprocess,sys;from pathlib import Path\nb=Path('+repr(base)+');payload=json.load(sys.stdin)\n'+r'''
assert not b.exists(),'Inspect the existing handle before retrying'
shutil.copytree('/dataset/shaoshitong/g1-common8-safety-cpu-d87fe08f9479',b,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
for name,value in payload['contents'].items():
 p=b/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value));assert hashlib.sha256(p.read_bytes()).hexdigest()==payload['sha256'][name]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
r=subprocess.run(['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-m','unittest','-v','search_g1.test_compute_precision','reproduction.common8.test_precision_evidence','reproduction.common8.test_model_smoke','reproduction.common8.test_configuration','reproduction.common8.test_safety_calibration_evidence'],cwd=b,env=env,text=True,capture_output=True,timeout=120)
x=dict(base=str(b),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256=payload['sha256']);(b/'explicit-precision-cpu-result.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),input=json.dumps(dict(contents=values,sha256=hashes)),text=True,capture_output=True,timeout=150)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(prep/'explicit-precision-cpu-result.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2));assert x['returncode']==0

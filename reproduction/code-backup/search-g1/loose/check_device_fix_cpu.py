import sys,json,shlex,subprocess,uuid,zlib,base64,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
repo=Path('/Users/shaoshitong/Documents/Search-G1/repo');prep=repo.parent/'preparation'
names=['verl/workers/fsdp_workers.py','verl/workers/actor/dp_actor.py','reproduction/common8/model_smoke.py','reproduction/common8/test_model_smoke.py'];files={n:(repo/n).read_text() for n in names};payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode();base='/dataset/shaoshitong/g1-common8-device-fix-cpu-'+uuid.uuid4().hex[:12]
code='import os,json,base64,zlib,hashlib,shutil,subprocess,time,inspect;from pathlib import Path\nb=Path('+repr(base)+');payload='+repr(payload)+'\n'+r'''
old=Path('/dataset/shaoshitong/Search-G1-common8');shutil.copytree(old/'repo',b,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
files=json.loads(zlib.decompress(base64.b64decode(payload)));hashes={}
for n,s in files.items():
 compile(s,n,'exec');(b/n).write_text(s);hashes[n]=hashlib.sha256((b/n).read_bytes()).hexdigest()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b));r=subprocess.run([str(old/'env/bin/python'),'-m','unittest','reproduction.common8.test_model_smoke','reproduction.common8.test_configuration'],cwd=b,env=env,capture_output=True,text=True)
print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256=hashes,base=str(b),scope='actual target CPU guard tests only; no GPU model or optimizer step proof',checked_unix=time.time())));assert r.returncode==0,r.stderr
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr;p=json.loads(r.stdout);(prep/'device-fix-cpu-proof.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))

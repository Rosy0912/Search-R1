import base64,gzip,hashlib,json,subprocess
from pathlib import Path
script="""export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import base64,gzip,hashlib,json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
out=root/'logs/policy-safety-calibration-52befe9-reviewed-binding-v7.json'
assert not out.exists(),'READY exists: inspect preserve'
cmd=[str(root/'env/bin/python'),str(root/'tools/policy-safety-calibration-controls-52befe9-v7/freeze_ready_v7.py'),str(root/'logs/CALIBRATION-CANDIDATE-52BEFE9-v7.json'),'b8f0436a2d1b9668664d6a6c8d14ba8c066d9f818c1b2c2a97eb2a420472ba4e',str(root/'logs/policy-safety-calibration-52befe9-physical-v7.json'),'2d0a19dd6ca1a55a13a481673fbe8a421339262b95ee4c1f2d323bb4b3e49a3e',str(root/'logs/CALIBRATION-DEPLOYMENT-INDEPENDENT-REVIEW-v7.json'),'7d49d6e1b1e4ef4fdfa18830a89921ae1d80c7cb459191678fe488a60f74f9f6',str(root/'logs/NATIVE3267-CALIBRATION-ACCEPTANCE-SPEC-v1.json'),str(out)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if out.exists():
 raw=out.read_bytes();r.update(path=str(out),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),gzip_base64=base64.b64encode(gzip.compress(raw)).decode())
print(json.dumps(r))
REMOTE
"""
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=360)
r={'transport_returncode':p.returncode,'transport_stderr':p.stderr}
if not p.returncode:
 a=json.loads(p.stdout);enc=a.pop('gzip_base64',None)
 if enc:
  raw=gzip.decompress(base64.b64decode(enc));assert hashlib.sha256(raw).hexdigest()==a['sha256']
  out=Path('outputs/gigpo-e1-policy-safety/policy-safety-calibration-52befe9-reviewed-binding-v7.json')
  with out.open('xb') as f:f.write(raw)
  a['local_original']=str(out.resolve())
 r['actual']=a
else:r['raw_stdout']=p.stdout
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V7-ACTUAL-FREEZE-v1.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'actual':r.get('actual')}))

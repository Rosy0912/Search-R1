import base64,gzip,hashlib,json,subprocess
from pathlib import Path
script="""export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/repo-policy-safety-52befe9:/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/tools/policy-safety-calibration-controls-52befe9-v8
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import base64,gzip,hashlib,json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
out=root/'logs/policy-safety-calibration-52befe9-physical-v8.json'
assert not out.exists(),'physical output exists: preserve inspect'
cmd=[str(root/'env/bin/python'),str(root/'tools/policy-safety-calibration-controls-52befe9-v8/verify_physical_v8.py'),str(root/'logs/CALIBRATION-CANDIDATE-52BEFE9-v8.json'),'e18615c998a69e0212a8aac99fb2a5db32556b97a98cdcfd05fcb87418b618d1',str(out)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if out.exists():
 raw=out.read_bytes();record.update(path=str(out),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),gzip_base64=base64.b64encode(gzip.compress(raw)).decode())
print(json.dumps(record))
REMOTE
"""
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=480)
r={'transport_returncode':p.returncode,'transport_stderr':p.stderr}
if not p.returncode:
 actual=json.loads(p.stdout);enc=actual.pop('gzip_base64',None)
 if enc:
  raw=gzip.decompress(base64.b64decode(enc));assert hashlib.sha256(raw).hexdigest()==actual['sha256']
  physical=Path('outputs/gigpo-e1-policy-safety/policy-safety-calibration-52befe9-physical-v8.json')
  with physical.open('xb') as f:f.write(raw)
  actual['local_original']=str(physical.resolve())
 r['actual']=actual
else:r['raw_stdout']=p.stdout
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V8-ACTUAL-PHYSICAL-v1.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'transport_returncode':p.returncode,'actual':r.get('actual')}))

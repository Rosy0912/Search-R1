"""Freeze exact independently accepted diagnostic deployment; never submit."""
import base64, gzip, hashlib, json, re, subprocess, sys
from pathlib import Path

candidate_sha, physical_sha, review_sha, helper_sha = sys.argv[1:5]
assert all(re.fullmatch('[0-9a-f]{64}', value) for value in (candidate_sha, physical_sha, review_sha, helper_sha))
script="""export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import base64,gzip,hashlib,json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
helper=root/'tools/policy-safety-calibration-controls-52befe9-v9/freeze_ready_v9.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='HELPER_SHA'
out=root/'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json'
assert not out.exists(),'Preserve existing READY; do not overwrite or retry'
cmd=[str(root/'env/bin/python'),str(helper),str(root/'logs/CALIBRATION-CANDIDATE-52BEFE9-v9.json'),
 'CANDIDATE_SHA',str(root/'logs/policy-safety-calibration-52befe9-physical-v9.json'),
 'PHYSICAL_SHA',str(root/'logs/CALIBRATION-V9-ACTUAL-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json'),
 'REVIEW_SHA',str(root/'logs/NATIVE3267-CALIBRATION-ACCEPTANCE-SPEC-v1.json'),str(out)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
if out.exists():
 raw=out.read_bytes();r.update(path=str(out),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),gzip_base64=base64.b64encode(gzip.compress(raw)).decode())
print(json.dumps(r))
REMOTE
"""
for key, value in [('CANDIDATE_SHA',candidate_sha),('PHYSICAL_SHA',physical_sha),('REVIEW_SHA',review_sha),('HELPER_SHA',helper_sha)]:
 script=script.replace(key,value)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=300)
r={'transport_returncode':p.returncode,'transport_stderr':p.stderr}
if not p.returncode:
 actual=json.loads(p.stdout);encoded=actual.pop('gzip_base64',None)
 if encoded:
  raw=gzip.decompress(base64.b64decode(encoded))
  assert hashlib.sha256(raw).hexdigest()==actual['sha256'] and len(raw)==actual['bytes']
  original=Path('outputs/gigpo-e1-policy-safety/policy-safety-calibration-52befe9-reviewed-binding-v9.json')
  with original.open('xb') as f:f.write(raw)
  actual['local_original']=str(original.resolve())
 r['actual']=actual
else:r['raw_stdout']=p.stdout
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V9-ACTUAL-FREEZE-v1.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                  'transport_returncode':p.returncode,'actual':r.get('actual')}))

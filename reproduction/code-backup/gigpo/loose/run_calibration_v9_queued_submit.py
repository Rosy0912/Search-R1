"""Exact reviewed v9 queued-once submission; never retries an uncertain claim."""
import hashlib, json, re, subprocess, sys
from pathlib import Path

binding_sha, review_sha = sys.argv[1:3]
assert all(re.fullmatch('[0-9a-f]{64}',value) for value in (binding_sha,review_sha))
local=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V9-ACTUAL-QUEUED-SUBMIT-v1.json')
assert not local.exists(),'Inspect existing submit evidence; do not execute again'
script="""export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
installed=json.loads((root/'logs/policy-safety-queued-dispatch-v9-install-receipt.json').read_text())
assert installed['status']=='EXACT_EXTERNAL_QUEUED_DISPATCHER_INSTALLED_NOT_SUBMITTED'
helper=root/'tools/policy-safety-calibration-queue-dispatch-v9/submit_queued_v9_once_v1.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='23e1e25d60689f9d8730dfd0e4f21167a7007498bf8f608753e08cae7a818950'
cmd=[str(root/'env/bin/python'),str(helper),str(root/'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json'),
 'BINDING_SHA',str(root/'logs/CALIBRATION-V9-QUEUED-SUBMIT-INDEPENDENT-REVIEW-v1.json'),'REVIEW_SHA']
p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
r['records']={}
for name in ['policy-safety-calibration-52befe9-submit-intent-v9.json','policy-safety-calibration-52befe9-submit-receipt-v9.json','policy-safety-calibration-52befe9-queue-submit-intent-v9.json','policy-safety-calibration-52befe9-queue-submit-receipt-v9.json']:
 f=root/'logs'/name
 if f.exists():
  raw=f.read_bytes();r['records'][name]=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=json.loads(raw))
print(json.dumps(r))
REMOTE
""".replace('BINDING_SHA',binding_sha).replace('REVIEW_SHA',review_sha)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=220)
r={'transport_returncode':p.returncode,'transport_stderr':p.stderr}
if not p.returncode:r['actual']=json.loads(p.stdout)
else:r['raw_stdout']=p.stdout
with local.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(output=str(local),sha256=hashlib.sha256(local.read_bytes()).hexdigest(),data=r)))

import datetime,hashlib,json,subprocess
from pathlib import Path
script="""export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import hashlib,json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
installer=root/'tools/install_calibration_v7.py'
assert hashlib.sha256(installer.read_bytes()).hexdigest()=='40e8aada7fb509d6d9d9b664eb1206bd684998f3f07e25f89a26398e71b2e57c'
cmd=[str(root/'env/bin/python'),str(installer),'--archive',str(root/'tools/policy-safety-calibration-52befe9-v7-package.tar.gz')]
results=[]
for mode in ['--verify-only','--install']:
 p=subprocess.run(cmd+[mode],capture_output=True,text=True,timeout=180)
 results.append({'mode':mode,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 if p.returncode:break
records={}
for name in ['policy-safety-calibration-52befe9-install-intent-v7.json','policy-safety-calibration-52befe9-install-receipt-v7.json']:
 f=root/'logs'/name
 if f.exists():
  raw=f.read_bytes();records[name]={'path':str(f),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results,'records':records}))
REMOTE
"""
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=400)
result={'transport_returncode':p.returncode,'transport_stderr':p.stderr,'raw_stdout':p.stdout}
if not p.returncode:result['actual']=json.loads(p.stdout)
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V7-ACTUAL-INSTALL-v1.json')
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'transport_returncode':p.returncode,'modes':[(r['mode'],r['returncode']) for r in result.get('actual',{}).get('results',[])],'receipt':result.get('actual',{}).get('records',{}).get('policy-safety-calibration-52befe9-install-receipt-v7.json',{}).get('sha256')}))
if p.returncode:print(p.stderr)

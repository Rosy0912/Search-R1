"""Install exact reviewed v9 metadata package once; does not submit or initialize CUDA."""
import hashlib, json, re, subprocess, sys
from pathlib import Path

installer_sha, package_sha = sys.argv[1:3]
assert all(re.fullmatch('[0-9a-f]{64}', value) for value in (installer_sha, package_sha))
script = """export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import hashlib,json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
installer=root/'tools/install_calibration_v9.py'
archive=root/'tools/policy-safety-calibration-52befe9-v9-package.tar.gz'
assert hashlib.sha256(installer.read_bytes()).hexdigest()=='INSTALLER_SHA'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='PACKAGE_SHA'
assert not (root/'logs/policy-safety-calibration-52befe9-install-intent-v9.json').exists(),'Inspect existing installation intent; never repeat uncertain mutation'
cmd=[str(root/'env/bin/python'),str(installer),'--archive',str(archive)]
results=[]
for mode in ['--verify-only','--install']:
 p=subprocess.run(cmd+[mode],capture_output=True,text=True,timeout=180)
 results.append({'mode':mode,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 if p.returncode:break
records={}
for name in ['policy-safety-calibration-52befe9-install-intent-v9.json','policy-safety-calibration-52befe9-install-receipt-v9.json']:
 f=root/'logs'/name
 if f.exists():
  raw=f.read_bytes();records[name]={'path':str(f),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results,'records':records}))
REMOTE
""".replace('INSTALLER_SHA', installer_sha).replace('PACKAGE_SHA', package_sha)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=400)
result={'transport_returncode':p.returncode,'transport_stderr':p.stderr,'raw_stdout':p.stdout}
if not p.returncode:result['actual']=json.loads(p.stdout)
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V9-ACTUAL-INSTALL-v1.json')
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                  'transport_returncode':p.returncode,
                  'modes':[(r['mode'],r['returncode']) for r in result.get('actual',{}).get('results',[])],
                  'receipt':result.get('actual',{}).get('records',{}).get('policy-safety-calibration-52befe9-install-receipt-v9.json',{}).get('sha256')}))
if p.returncode:print(p.stderr)

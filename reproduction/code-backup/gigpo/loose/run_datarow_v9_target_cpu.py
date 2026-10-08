"""New v9 driver, inherited actual DataProto CPU-only test; never initializes GPU/Ray."""
import base64, hashlib, json, re, subprocess, sys
from pathlib import Path

driver_sha = sys.argv[1]
assert re.fullmatch('[0-9a-f]{64}', driver_sha)
script = """export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /usr/bin/python3 - <<'REMOTE'
import base64,hashlib,json,os,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
probe=root/'tools/policy-safety-datarow-v9-cpu-probe'
driver=probe/'calibration_driver_v1.py'
test=probe/'test_calibration_datarow_v8_actual.py'
assert hashlib.sha256(driver.read_bytes()).hexdigest()=='DRIVER_SHA'
assert hashlib.sha256(test.read_bytes()).hexdigest()=='dd379a3a50f78a49ba21b8ef805c9a7b7fecf4eba90a9832dca6ea4f37ba87b9'
out=root/'logs/CALIBRATION-DATAROW-TARGET-CPU-v9.json'
assert not out.exists(),'Preserve existing proof; do not rerun/overwrite'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
         PYTHONPATH=str(root/'repo-policy-safety-52befe9'))
p=subprocess.run([str(root/'env/bin/python'),str(test),'--runtime',str(probe),'--json-out',str(out)],
                 cwd=str(probe),env=env,capture_output=True,text=True,timeout=180)
records={}; encoded=None
if out.exists():
 raw=out.read_bytes();records['target_cpu']={'path':str(out),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)}
 encoded=base64.b64encode(raw).decode()
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,
                  'stdout':p.stdout,'stderr':p.stderr,'records':records,'cpu_base64':encoded}))
REMOTE
""".replace('DRIVER_SHA', driver_sha)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=250)
result={'transport_returncode':p.returncode,'transport_stderr':p.stderr,'raw_stdout':p.stdout}
if not p.returncode:
 result['actual']=json.loads(p.stdout)
 encoded=result['actual'].pop('cpu_base64')
 if encoded is not None:
  raw=base64.b64decode(encoded)
  actual_record=result['actual']['records']['target_cpu']
  assert len(raw)==actual_record['bytes'] and hashlib.sha256(raw).hexdigest()==actual_record['sha256']
  with Path('outputs/gigpo-e1-policy-safety/CALIBRATION-DATAROW-TARGET-CPU-v9.json').open('xb') as f:f.write(raw)
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-DATAROW-TARGET-CPU-RAW-v9.json')
with out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
actual=result.get('actual',{})
record=actual.get('records',{}).get('target_cpu')
if record:
 print(json.dumps({'raw_wrapper':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                   'transport_returncode':p.returncode,'test_returncode':actual.get('returncode'),'target_record':record}))
else:
 print(json.dumps({'raw_wrapper':str(out),'transport_returncode':p.returncode,'actual':actual}))

"""Reuse reviewed read-only scheduler observer for the one actual submitted v9 job."""
import hashlib, json, subprocess, sys
from pathlib import Path
job=sys.argv[1];assert job.isdigit()
script="""/usr/bin/python3 - <<'REMOTE'
import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
job='JOB_ID'
receipt=json.loads((root/'logs/policy-safety-calibration-52befe9-submit-receipt-v9.json').read_text())
assert receipt['status']=='SUBMITTED_EXACTLY_ONCE' and receipt['job_id']==job
helper=root/'tools/watch_policy_safety_calibration_terminal_v1.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='2c244d4636ec6230dd8d9c9296e14007cad07ca5b3d89f7e7ef57ea7c5307515'
claim=root/('logs/policy-safety-calibration'+job+'-watch-claim-v1.json')
start=root/('logs/policy-safety-calibration'+job+'-watch-start-v1.json')
assert not claim.exists() and not start.exists(),'Existing watch intent; inspect instead of duplicate'
with claim.open('x') as f:
 json.dump(dict(job=job,metadata_only=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f);f.flush();os.fsync(f.fileno())
stdout=root/('logs/policy-safety-calibration'+job+'-watch-stdout-v1.log')
stderr=root/('logs/policy-safety-calibration'+job+'-watch-stderr-v1.log')
with stdout.open('x') as so,stderr.open('x') as se:
 p=subprocess.Popen(['/usr/bin/python3',str(helper),job],stdout=so,stderr=se,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,pid=p.pid,poll=p.poll(),helper=str(helper),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),claim_sha256=hashlib.sha256(claim.read_bytes()).hexdigest(),metadata_only=True,terminal_expected=str(root/('logs/policy-safety-calibration-terminal-'+job+'-raw-v1.json')))
 try:r['proc_stat']=Path('/proc/'+str(p.pid)+'/stat').read_text()
 except Exception as error:r['proc_stat_error']=repr(error)
 with start.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(path=str(start),sha256=hashlib.sha256(start.read_bytes()).hexdigest(),data=r)))
REMOTE
""".replace('JOB_ID',job)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=180)
r=dict(transport_returncode=p.returncode,transport_stderr=p.stderr,raw_stdout=p.stdout)
if not p.returncode:r['actual']=json.loads(p.stdout)
out=Path('outputs/gigpo-e1-policy-safety')/('CALIBRATION'+job+'-WATCH-START-v1.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),data=r)))

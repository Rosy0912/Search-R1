import datetime, hashlib, json, subprocess
from pathlib import Path

script = """/usr/bin/python3 - <<'REMOTE'
import datetime,json,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=35)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node='h100-8',read_only=True,
 slurm=run(['squeue','-w','h100-8','-h','-o','%i|%T|%j|%u|%R']),
 gpu=run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']),
 compute=run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']))
r['v8_submission_paths']={str(root/'logs'/name):(root/'logs'/name).exists() for name in ['policy-safety-calibration-52befe9-submit-intent-v8.json','policy-safety-calibration-52befe9-submit-receipt-v8.json']}
r['v9_submission_paths']={str(root/'logs'/name):(root/'logs'/name).exists() for name in ['policy-safety-calibration-52befe9-submit-intent-v9.json','policy-safety-calibration-52befe9-submit-receipt-v9.json']}
print(json.dumps(r))
REMOTE
"""
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=180)
r=dict(transport_returncode=p.returncode,transport_stderr=p.stderr)
if p.returncode==0:
 r['actual']=json.loads(p.stdout)
else:
 r['raw_stdout']=p.stdout
name='CALIBRATION-V9-RESOURCE-READONLY-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
out=Path('outputs/gigpo-e1-policy-safety')/name
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),data=r)))

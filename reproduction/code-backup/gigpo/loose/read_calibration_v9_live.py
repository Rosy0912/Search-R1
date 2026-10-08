"""Bounded read-only live metadata for the job identified by actual v9 submit receipt."""
import datetime, hashlib, json, subprocess, sys
from pathlib import Path
job=sys.argv[1];assert job.isdigit()
script="""/usr/bin/python3 - <<'REMOTE'
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004');job='JOB_ID'
receipt=json.loads((root/'logs/policy-safety-calibration-52befe9-submit-receipt-v9.json').read_text())
assert receipt['status']=='SUBMITTED_EXACTLY_ONCE' and receipt['job_id']==job
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=25)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node='h100-8',job_id=job,
 source='ACTUAL_V9_SUBMIT_RECEIPT',receipt=receipt,
 scheduler=cmd(['scontrol','show','job',job,'-o']),queue=cmd(['squeue','-w','h100-8','-h','-o','%i|%T|%j|%N']),
 gpu=cmd(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']),
 compute=cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader']))
runs=list((root/'runs').glob('gigpo-e1-policy-safety-calibration25-job'+job+'-*'))
assert len(runs)<=1,'Multiple real runs for once-submitted job'
r['actual_run']=str(runs[0]) if runs else None;r['originals']=[];r['log_tails']=[]
if runs:
 run=runs[0]
 for name in ['run-result.json','calibration-state.json','actual-vllm-backend-identity.json','restore25-before-baseline.json','restore25-before-rollout.json','baseline-evaluation.json','actual-probability-alignment.json','probability-material-manifest.json','probability-identity-ready.json','post-sampling-pause-state.json','ray-audit-environment-verification.json']:
  p=run/name
  if p.exists():
   raw=p.read_bytes();s=p.stat();record=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),inode=s.st_ino,mtime_ns=s.st_mtime_ns)
   if len(raw)<1024*1024:record['data']=json.loads(raw)
   r['originals'].append(record)
 logs=[root/('logs/policy-safety-calibration-'+job+'.log'),run/'calibration.log',run/'guard.log']
 sessions=[p for p in (run/'ray').glob('session_*') if p.is_dir() and not p.is_symlink()]
 for session in sessions:
  logs += [p for p in (session/'logs').glob('worker-*.out') if p.is_file()]
 for p in logs:
  if p.exists():
   s=p.stat()
   with p.open('rb') as f:f.seek(max(0,s.st_size-5000));tail=f.read().decode(errors='replace')
   r['log_tails'].append(dict(path=str(p),inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,last5000=tail))
terminal=root/('logs/policy-safety-calibration-terminal-'+job+'-raw-v1.json')
if terminal.exists():
 raw=terminal.read_bytes();r['terminal_original']=dict(path=str(terminal),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=json.loads(raw))
print(json.dumps(r))
REMOTE
""".replace('JOB_ID',job)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=180)
r=dict(transport_returncode=p.returncode,transport_stderr=p.stderr)
if not p.returncode:r['actual']=json.loads(p.stdout)
else:r['raw_stdout']=p.stdout
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=Path('outputs/gigpo-e1-policy-safety')/('CALIBRATION'+job+'-LIVE-'+stamp+'-v9.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),transport_returncode=p.returncode,
 scheduler=r.get('actual',{}).get('scheduler'),run=r.get('actual',{}).get('actual_run'),
 originals=[dict(path=x['path'],sha256=x['sha256']) for x in r.get('actual',{}).get('originals',[])])))

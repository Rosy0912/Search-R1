import sys,json,subprocess,shlex,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');h=json.loads((prep/'model-device-diagnostic-launch.json').read_text())
code='import json,subprocess,time;from pathlib import Path\nh='+repr(h)+'\n'+r'''
b=Path(h['base']);r=subprocess.run(['scontrol','show','job',h['job_id'],'-o'],capture_output=True,text=True)
p=b/'logs'/('slurm-'+h['job_id']+'.log');s=p.read_text(errors='replace') if p.exists() else ''
records=[]
for line in s.splitlines():
 for marker in ('COMMON8_DEVICE_DIAGNOSTIC=','COMMON8_BATCH_DEVICE_DIAGNOSTIC='):
  if marker in line:
   try:records.append(json.loads(line.split(marker,1)[1]))
   except ValueError:pass
out=dict(checked_unix=time.time(),job_stdout=r.stdout,job_stderr=r.stderr,records=records,log_tail=s[-5000:])
p=b/'logs/model-smoke-initial/complete.json'
if p.exists():out['complete']=json.loads(p.read_text())
print(json.dumps(out))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'model-device-diagnostic-observation.json').write_text(json.dumps(p,indent=2));print(json.dumps({k:v for k,v in p.items() if k!='log_tail'},indent=2))

"""One-shot read-only wait then exact original worker callback capture, not heartbeat."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(4,5),required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
assert os.getuid()==1001 and Path(a.output).parent==root/'receipts'
start=time.monotonic()
while time.monotonic()-start<7200:
 raw=Path('/proc/401228/stat').read_text();assert int(raw.rsplit(')',1)[1].split()[19])==651299742
 status=Path('/proc/401228/status').read_text();assert next(l for l in status.splitlines() if l.startswith('Uid:')).split()[1]=='1001'
 manifest=root/'checkpoints/training_state'/('global_step_%d'%a.step)/'manifest.json'
 if manifest.exists():raise ValueError('Checkpoint already committed; never attach a later stream')
 evidence=root/'output/rank-audit/policy-safety'
 if any(evidence.glob('step-%06d-*'%a.step)):
  print('QUEUE_AT_ACTUAL_UPDATE_STEP='+str(a.step),flush=True)
  rc=subprocess.call([sys.executable,'-B',str(a.capture),'--ray-address','10.123.4.12:62462','--step',str(a.step),'--output',a.output])
  sys.exit(rc)
 print(json.dumps(dict(time=time.time(),waiting_for_actual_actor_update=a.step)),flush=True)
 time.sleep(30)
raise TimeoutError('Capture wait ended without verified actor update')

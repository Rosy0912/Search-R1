"""Request durable HKUST3 archiving; never delete an unverified checkpoint."""
import json,time,os
from pathlib import Path

def request(root,step):
 root=Path(root);folder=root/'archive-requests';folder.mkdir(exist_ok=True)
 receipt=json.loads((root/'checkpoints'/('global_step_'+str(step))/'full-sha-receipt.json').read_text())
 (root/('checkpoint-receipt-'+str(step)+'.json')).write_text(json.dumps(receipt,indent=2))
 p=folder/(str(step)+'.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'step':step,'receipt':receipt}));os.replace(tmp,p)
 deadline=time.monotonic()+7200
 while not (root/('archive-verified-'+str(step)+'.json')).exists():
  if time.monotonic()>deadline:raise RuntimeError('HKUST3 archive did not verify; HPC checkpoint retained')
  time.sleep(10)
 result=json.loads((root/('archive-verified-'+str(step)+'.json')).read_text())
 assert result['status']=='HKUST3_ALL_CHECKPOINT_SHA_PASS' and result['step']==step
 return result

def allow_cleanup(root,step):
 (Path(root)/('archive-delete-safe-'+str(step))).write_text('save/load/validation uses complete\n')

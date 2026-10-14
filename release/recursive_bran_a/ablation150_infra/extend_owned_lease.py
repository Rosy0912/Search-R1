"""Extend only the allocation already running this explicitly authorized study."""
import json,subprocess,time
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-ablation150-20261014')
e=json.loads((b/'C1-root-only/receipts/current-entry.json').read_text())
assert e['run_id']=='260d1debb6214d75b3d1a5954d88f524' and e['allocation']=='3393'
p=Path('/proc',str(e['pid']));assert p.stat().st_uid==e['uid']
assert int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==e['tick']
before=subprocess.check_output(['scontrol','show','job','3393'],text=True)
assert 'JobState=RUNNING' in before and 'NodeList=h100-11' in before and 'UserId=root(0)' in before
limit=subprocess.check_output(['scontrol','show','partition','gpu'],text=True)
assert 'MaxTime=UNLIMITED' in limit
r=subprocess.run(['scontrol','update','JobId=3393','TimeLimit=14-00:00:00'],text=True,capture_output=True)
after=subprocess.check_output(['scontrol','show','job','3393'],text=True)
print(json.dumps(dict(time=time.time(),rc=r.returncode,error=r.stderr,before=before,after=after,
 scope='Existing exclusive owned allocation only, duration for seven authorized sequential150 groups; no new GPU, process signal, model, or scientific parameter change')),flush=True)
assert r.returncode==0 and 'TimeLimit=14-00:00:00' in after

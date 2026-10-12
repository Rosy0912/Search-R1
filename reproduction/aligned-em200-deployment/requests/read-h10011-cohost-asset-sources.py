"""Read existing cross-node file visibility without running GPU code or copying data."""
import os,json,datetime,subprocess
from pathlib import Path
assert os.uname().nodename=='h100-11'
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'scope':'READONLY_LOCALIZATION_INPUT_DISCOVERY'}
paths=[Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets'),Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/assets')]
for base in paths:
 entry={'exists':base.exists(),'children':[]}
 if base.is_dir():
  for p in sorted(base.iterdir()):
   d={'name':p.name,'link':p.is_symlink(),'resolved':str(p.resolve()),'exists':p.exists()}
   if p.is_dir():d['children']=[{'name':q.name,'bytes':q.stat().st_size,'link':q.is_symlink(),'resolved':str(q.resolve())} for q in sorted(p.iterdir())]
   entry['children'].append(d)
 r['files'][str(base)]=entry
p=subprocess.run(['which','rsync'],capture_output=True,text=True,timeout=5);r['rsync']=p.stdout.strip()
print(json.dumps(r))

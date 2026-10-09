import json,hashlib,base64
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009');run=r/'runs/formal-aligned-em200-ecbddc0404f041babd8fef3b3a9d7a1e-3358';out={}
for name,p in [('cpu',r/'evidence/first5-cpu3358-20261009-1806/cpu-evidence.json'),('manifest',run/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json')]:
 if p.exists():
  raw=p.read_bytes();assert len(raw)<16000000;out[name]={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps(out))

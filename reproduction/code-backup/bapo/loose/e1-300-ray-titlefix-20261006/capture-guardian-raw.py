import json,base64,hashlib
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006');files={}
for n in ['logs/guardian-tests-durable.log','logs/guardian-tests-durable-exit.json','logs/tools-manifest.json']:
 b=(r/n).read_bytes();files[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps({'root':str(r),'files':files}))

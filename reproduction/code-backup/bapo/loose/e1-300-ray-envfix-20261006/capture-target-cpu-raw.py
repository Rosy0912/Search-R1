import base64,hashlib,json
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');files={}
for n in ['logs/cpu-gate-durable.log','logs/cpu-gate-durable-exit.json','logs/guardian-tests-durable.log','logs/guardian-tests-durable-exit.json','logs/runtime-executable.json','logs/runtime-capture.log','logs/tools-manifest.json','logs/title-dependency-preflight.json','logs/title-dependency-preflight.log','logs/title-dependency-preflight-exit.json','source-expected.txt']:
 b=(r/n).read_bytes();files[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps({'root':str(r),'files':files}))
